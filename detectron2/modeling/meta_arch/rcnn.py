# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.refdet_components.uniter.match_grd import UniterForNegRE

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from detectron2.modeling.refdet_heads.RCRN import build_refdet_head


__all__ = ["GeneralizedRCNN", "ProposalNetwork", "OurArc"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class OurArc(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg, pretrained_opts):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # load uniter
        if cfg.MODEL.REF.VIS_FEAT_TYPE=='uniter':
            self.uniter_modal = 'bi'
            self.uniter_logits = 'none'
            opts=pretrained_opts
            if opts.checkpoint:
                checkpoint = torch.load(opts.checkpoint)
            else:
                checkpoint = {}
            
            self.uniter=UniterForNegRE.from_pretrained(
                opts.model_config, checkpoint,
                img_dim=2048, loss=opts.train_loss,
                margin=opts.margin,
                hard_ratio=opts.hard_ratio, mlp=opts.mlp, pool=opts.pool, 
                pool_with_fuse=opts.pool_with_fuse, ng_branch=opts.ng_branch,
                language_only=opts.language_only,match_only=opts.match_only,
                use_prompt=opts.use_prompt,itm_match_head=opts.itm_match_head,
                opts=opts, use_layer=cfg.MODEL.REF.USE_LAYER,fix_layer=cfg.MODEL.REF.FIX_LAYER)
            self.extra_layer=cfg.MODEL.REF.EXTRA_LAYER
        else:
            self.uniter=None

        self.refdet_head = build_refdet_head(cfg)
        self.head_modules = None

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.as_tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.as_tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs, *args, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, training_mode=False)

        if self.uniter is not None:
            
            uniter_res = self.uniter(
                batched_inputs,compute_loss=False, extract_feat=True,
                extract_layer=self.extra_layer,modal=self.uniter_modal,
                device=self.device,compute_logits=self.uniter_logits
                )
            
            img_hidden,txt_hidden,match_logits,grd_logits=[uniter_res[k] for k in ['img_hidden','txt_hidden','match_logits','grd_logits']]
            batched_inputs['roi_gt'][2]=img_hidden
            
            
            if self.uniter_logits!='none':
                batched_inputs['match_logits']=match_logits[:,0].unsqueeze(1)
                # grd_logits:bs*maxnum_bb
                batched_inputs['grd_logits']=self.padding_grd_logits(grd_logits,self.device)
            
            
            merged_txt_hidden=self.batch_merge_emb(txt_hidden,batched_inputs['token_split_num_list'])
            phrase_feat=self.phrase_hiddens(merged_txt_hidden,batched_inputs['phrasenum_idx'],batched_inputs['entity_spo_positions'],self.device)
            batched_inputs['phrase_feat']=phrase_feat

        
        all_loss, results = self.refdet_head(batched_inputs, self.device)
        # loss: dict()
        # results: dict(ncorrect, ntotal, pred_ind, pred_coord)

        return all_loss, results
    
    def merge_emb_per_sent(self,sent_embs,split_list):
        embs_len=len(sent_embs)# with padding
        split_list.append(embs_len-sum(split_list)) # split padding embs
        embs=list(torch.split(sent_embs,split_list))
        
        embs_no_padding=[i.mean(dim=0,keepdim=True) for i in embs[:-1]]
        sent_hidden=torch.cat(embs_no_padding)
        return sent_hidden

    def batch_merge_emb(self,txt_hidden,split_lists):
        sent_embs_list=[]
        for sent_hidden,split_list in zip(txt_hidden,split_lists):
            merge_sent_hidden=self.merge_emb_per_sent(sent_hidden,split_list)
            sent_embs_list.append(merge_sent_hidden)
        batch_txt_hidden=pad_sequence(sent_embs_list,batch_first=True)
        return batch_txt_hidden

    def padding_grd_logits(self,grd_logits,device):
        bs,max_num_bb=grd_logits.size()
        pad_logits=torch.cat((grd_logits, -1*torch.ones(bs,126-max_num_bb).to(device)),1)
        return pad_logits

    def phrase_hiddens(self, txt_hidden, phrasenum_idx, positions, device):
        
        if len(txt_hidden.size())!=3:
            txt_hidden=txt_hidden.unsqueeze(0) 
        txt_hidden=torch.cat([txt_hidden,torch.zeros(txt_hidden.size(0),1,txt_hidden.size(-1)).to(device)],1)
        return txt_hidden[phrasenum_idx,positions]

    def inference(self, batched_inputs, *args, training_mode=False, **kwargs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        if self.uniter is not None:
            uniter_res = self.uniter(
                batched_inputs,compute_loss=False, extract_feat=True,
                extract_layer=self.extra_layer,modal=self.uniter_modal,
                device=self.device,compute_logits=self.uniter_logits
                )
            img_hidden,txt_hidden,match_logits,grd_logits,all_layer_atts=[uniter_res[k] for k in ['img_hidden','txt_hidden','match_logits','grd_logits','all_layer_atts']]    
            batched_inputs['roi_gt'][2]=img_hidden

            if all_layer_atts is not None:
                batched_inputs['all_layer_atts']=all_layer_atts

            if self.uniter_logits!='none':
                batched_inputs['match_logits']=match_logits[:,0].unsqueeze(1)
                # grd_logits:bs*maxnum_bb
                batched_inputs['grd_logits']=self.padding_grd_logits(grd_logits,self.device)

            merged_txt_hidden=self.batch_merge_emb(txt_hidden,batched_inputs['token_split_num_list'])
            phrase_feat=self.phrase_hiddens(merged_txt_hidden,batched_inputs['phrasenum_idx'],batched_inputs['entity_spo_positions'],self.device)
            batched_inputs['phrase_feat']=phrase_feat
            
        all_losses, results = self.refdet_head(batched_inputs, self.device, training_mode=training_mode)
        for k,v in all_losses.items():
            results[k] = v.item()
        return results  # dict(ncorrect, ntotal, pred_ind, pred_coord, loss)

    def train(self, mode=True):
        super().train(mode)
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        return self