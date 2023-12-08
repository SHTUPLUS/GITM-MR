# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import os
import pickle as pkl
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from fvcore.common.file_io import PathManager
from scipy.special import softmax

import detectron2.utils.comm as comm
import detectron2.utils.refdet_basics as basics
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from .evaluator import DatasetEvaluator


class RefdetEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        # self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._predictions = None

        metadata = MetadataCatalog.get(dataset_name)

        self.colors = metadata.thing_colors
        self.save_img = cfg.EVAL_SAVE_IMAGE
        self.img_dump_dir = os.path.join(self._output_dir, 'img_results')
        os.makedirs(self.img_dump_dir, exist_ok=True)

        self.data_root = './data'
        split = 'val' if 'val' in dataset_name else 'test'

        self.exps = json.load(open(os.path.join(self.data_root, 'expression', '{}_expressions_ori_rel.json'.format(split)), 'r'))
        self.exps.update(json.load(open(os.path.join(self.data_root, 'expression', '{}_expressions_foil_rel.json'.format(split)), 'r')))

        self.max_acc = 0
        self.info_list = list()

        if 'len' in cfg.DATASETS.TRAIN[0]:
            idx = cfg.DATASETS.TRAIN[0].find('len')
            self.ood = int(cfg.DATASETS.TRAIN[0][idx + 3:idx + 5])
        else:
            self.ood = 0

        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        self._predictions = None
        self.info_list = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs:
            padded_boxcoord, padded_boxcat, padded_roifeat, padded_locfeat, padded_relloc_table,
               padded_word_idx_seq, gt_idx, gt_xyxy, dataset_dict['image_id'], dataset_dict['sent_id']
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # outputs is a dict, each element is of batch size
        if self._predictions is None:
            self._predictions = {k: [v] for k, v in outputs.items()}
        else:
            for k, v in self._predictions.items():
                v.append(outputs[k])  # list of bs,.. ndarrays

        preds, gts, imgids, sentids, unique_ids, levels = [outputs[k] for k in
                                                           ['pred_coord', 'gt_coord', 'img_ids', 'sent_ids',
                                                            'unique_ids', 'full_path_lengths']]
        pred_ind, gt_ind, sorted_rois, sorted_raw_scores = [outputs[k] for k in
                                                            ['pred_ind', 'gt_ind', 'sorted_rois', 'sorted_scores']]

        counter_type, matchlist = [outputs[k] for k in ['counter_type', 'match']]
        pred_matchlist = outputs['pred_match']
        match_correct_or_not = pred_matchlist == matchlist

        pred_foil = outputs['pred_foil']
        gt_foil = outputs['gt_foil']

        correct_or_not = np.asarray([basics.compute_IoU(pred_c, gt) > 0.5 for pred_c, gt in zip(preds, gts)])

        gt_color, pred_color = self.colors
        for idx, (imgid, gt, pred, level, sentid, uniqueid, pred_i, gt_i, rois, scores, correct) in \
                enumerate(zip(imgids, gts, preds, levels, sentids, unique_ids, pred_ind, gt_ind, sorted_rois,
                              sorted_raw_scores, correct_or_not)):

            not_padded_ind = np.logical_not(np.all(rois == 0, axis=1))  # should be not (all(0)), not all(not(0))
            max_num_top = 5
            num_top = min(max_num_top, not_padded_ind.sum())
            not_padded_rois = rois[not_padded_ind]
            not_padded_scores = scores[not_padded_ind]
            normalized_scores = softmax(not_padded_scores)

            if self.save_img and matchlist[idx]:
                imgpath = os.path.join(self.data_root, 'images', f'{imgid}.jpg')
                im = cv2.imread(imgpath)[:, :, ::-1]
                vis = Visualizer(im, None)

                boxes = np.concatenate((gt[None], rois[:num_top]), axis=0).tolist()
                tmp = ['{}-{:.4}'.format(idx + 1, s) for idx, s in enumerate(normalized_scores[:num_top].round(4))]
                labels = ['gt'] + tmp
                colors = [gt_color] + [pred_color] * num_top
                result = vis.overlay_instances(boxes=boxes,
                                               labels=labels,
                                               assigned_colors=colors)
                im = result.get_image()
                # save image
                with open(os.path.join(self.img_dump_dir, 'sentid_{}.pkl'.format(sentid)), 'wb') as f:
                    pkl.dump(im, f)

            tostore = dict(level=level, imgid=imgid, sentid=sentid, uniqueid=uniqueid, correct=correct,
                           roicoord=not_padded_rois, after_scores=normalized_scores, before_scores=not_padded_scores,
                           pred_ind=pred_ind, gt_ind=gt_ind, gtcoord=gt, gt_i=gt_i,
                           pred_i=pred_i)  # correct only considers pred and gt

            tostore['match'] = matchlist[idx]
            tostore['counter_type'] = counter_type[idx]

            tostore['matchcorrect'] = matchcorrect = match_correct_or_not[idx]
            tostore['pred_foil'] = pred_foil
            tostore['gt_foil'] = gt_foil

            if self.save_img and matchlist[idx]:
                save_name = f'stats_{uniqueid}.pkl'
                # save stats
                with open(os.path.join(self.img_dump_dir, save_name), 'wb') as f:
                    pkl.dump(tostore, f)

            item_info = [sentid, imgid, uniqueid, level, correct, normalized_scores, not_padded_scores]
            item_info.append(matchcorrect)
            item_info += [matchlist[idx], counter_type[idx]]
            self.info_list.append(item_info)

    def evaluate(self):
        keys = ['ncorrect', 'ntotal', 'pred_ind', 'gt_ind',
                'pred_coord', 'gt_coord', 'img_ids', 'sent_ids', 'unique_ids', 'full_path_lengths', 'sorted_scores', ]
        keys += ['loss']

        keys += ['match', 'counter_type', 'num_all']  # num_all is number of all examples
        keys += ['pred_match', 'num_matchcorrect', 'match_loss']

        keys += ['pred_foil', 'gt_foil']

        if self._distributed:
            sync_st = time.time()
            comm.synchronize()

            predictions = comm.gather(self._predictions, dst=0)
            info_list = comm.gather(self.info_list, dst=0)
            if not comm.is_main_process():
                return {}
            else:
                if 'prob_r_pairs' in predictions[0]: keys.append('prob_r_pairs')

            final_preds = OrderedDict()
            for k in keys:
                if k not in predictions[0]:
                    print(k, 'not in pred')
                    continue
                tmp = list(
                    itertools.chain(*[pred[k] for pred in predictions]))  # list of list of tuples => list of tuples
                if isinstance(tmp[0], np.ndarray):
                    final_preds[k] = np.concatenate(tmp)
                elif isinstance(tmp[0], tuple):
                    final_preds[k] = list(itertools.chain(*tmp))  # list of list of tuple
                else:
                    final_preds[k] = np.asarray(tmp)

            # save brief basic infos
            info_list = list(itertools.chain(*info_list))
            with open(os.path.join(self.img_dump_dir, f'stats.pkl'), 'wb') as f:
                pkl.dump(info_list, f)
            self._logger.info("sync pred takes {} sec".format(round(time.time() - sync_st)))

        else:
            keys = list(self._predictions.keys())
            final_preds = {}
            for k in keys:
                final_preds[k] = np.concatenate(self._predictions[k])
        print(keys)

        gt_coord = np.asarray(final_preds['gt_coord']).reshape((-1, 4))
        pred_coord = np.asarray(final_preds['pred_coord']).reshape((-1, 4))

        match_target = np.asarray(final_preds['match'])
        match_pred = np.asarray(final_preds['pred_match']).reshape((-1,))
        tp = np.logical_and(match_pred == match_target, match_target)
        tf = np.logical_and(match_pred == match_target, np.logical_not(match_target))

        grd_nc = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])[tp])
        grd_nc_ora = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])[
                             match_target])
        grd_nc = int(grd_nc)
        grd_nc_ora = int(grd_nc_ora)

        len_dict = self._get_length_in_batch(final_preds['sent_ids'])
        ood_list = len_dict > self.ood
        grd_nc_in = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])
                        [np.logical_and(tp, np.logical_not(ood_list))])
        grd_nt_in = sum(match_target[np.logical_not(ood_list)])
        grd_nc_ood = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])
                         [np.logical_and(tp, ood_list)])
        grd_nt_ood = sum(match_target[ood_list])
        grd_nc_in_ora = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])
                            [np.logical_and(match_target, np.logical_not(ood_list))])
        grd_nc_ood_ora = sum(torch.tensor(
            [basics.compute_IoU(pred_c, gt_c) > 0.5 for pred_c, gt_c in zip(pred_coord, gt_coord)])
                             [np.logical_and(match_target, ood_list)])

        pos_nt = sum(match_target)
        neg_bool = np.logical_not(match_target)
        grdonly_acc = grd_nc / float(pos_nt) * 100
        grd_acc_ora = grd_nc_ora / float(pos_nt) * 100

        grd_acc_in = float(grd_nc_in) / float(grd_nt_in) * 100
        grd_acc_in_ora = float(grd_nc_in_ora) / float(grd_nt_in) * 100
        grd_acc_ood = float(grd_nc_ood) / float(grd_nt_ood) * 100
        grd_acc_ood_ora = float(grd_nc_ood_ora) / float(grd_nt_ood) * 100


        match_nc = (match_target == match_pred).sum()
        neg_nt = sum(neg_bool)
        match_nt = len(match_target)
        match_acc = match_nc / float(match_nt) * 100
        self._logger.info('matchcls_acc={}%={}/{}'.format(round(match_acc, 2), match_nc, match_nt))

        # pos_nc = (match_target == match_pred)[match_target].sum()
        # neg_nc = (match_target == match_pred)[neg_bool].sum()
        # pos_acc = pos_nc / float(pos_nt) * 100
        # neg_acc = neg_nc / float(neg_nt) * 100
        # self._logger.info('matchpos_acc={}%={}/{}'.format(round(pos_acc, 2), pos_nc, pos_nt))
        # self._logger.info('matchneg_acc={}%={}/{}'.format(round(neg_acc, 2), neg_nc, neg_nt))

        if self.ood:
            match_nc_ood = (match_target == match_pred)[ood_list].sum()
            match_nt_ood = len(match_target[ood_list])
            match_acc_ood = match_nc_ood / match_nt_ood * 100
            match_nc_in = (match_target == match_pred)[np.logical_not(ood_list)].sum()
            match_nt_in = len(match_target[np.logical_not(ood_list)])
            match_acc_in = match_nc_in / match_nt_in * 100
            self._logger.info(
                'matchcls_acc_in={}%={}/{}'.format(round(match_acc_in, 2), match_nc_in, match_nt_in))
            self._logger.info(
                'matchcls_acc_ood={}%={}/{}'.format(round(match_acc_ood, 2), match_nc_ood, match_nt_ood))

        acc = match_acc

        self._logger.info('grndonly_acc={}%={}/{}'.format(round(grdonly_acc, 2), grd_nc, pos_nt))
        self._logger.info('grndora_acc={}%={}/{}'.format(round(grd_acc_ora, 2), grd_nc_ora, pos_nt))
        if self.ood:
            self._logger.info('grndonly_in_acc={}%={}/{}'.format(round(grd_acc_in, 2), grd_nc_in, grd_nt_in))
            self._logger.info(
                'grndonly_in_acc_ora={}%={}/{}'.format(round(grd_acc_in_ora, 2), grd_nc_in_ora, grd_nt_in))
            self._logger.info('grndonly_out_acc={}%={}/{}'.format(round(grd_acc_ood, 2), grd_nc_ood, grd_nt_ood))
            self._logger.info(
                'grndonly_out_acc_ora={}%={}/{}'.format(round(grd_acc_ood_ora, 2), grd_nc_ood_ora, grd_nt_ood))

        pred_foil = final_preds['pred_foil']
        gt_foil = final_preds['gt_foil']
        correct_edge = (gt_foil == pred_foil)[neg_bool].sum()
        foil_edge_acc = correct_edge / float(neg_nt) * 100
        self._logger.info('foil_acc_ora={}%={}/{}'.format(round(foil_edge_acc, 2), correct_edge, neg_nt))

        if self.ood:
            correct_edge = (gt_foil == pred_foil)[tf].sum()
            foil_edge_acc = correct_edge / float(neg_nt) * 100
            self._logger.info('foil_acc={}%={}/{}'.format(round(foil_edge_acc, 2), correct_edge, neg_nt))

            correct_edge_in = (gt_foil == pred_foil)[np.logical_and(tf, np.logical_not(ood_list))].sum()
            neg_nt_in = sum(neg_bool[np.logical_not(ood_list)])
            foil_edge_acc_in = correct_edge_in / float(neg_nt_in) * 100
            self._logger.info(
                'foil_acc_in={}%={}/{}'.format(round(foil_edge_acc_in, 2), correct_edge_in, neg_nt_in))

            correct_edge_in = (gt_foil == pred_foil)[
                np.logical_and(neg_bool, np.logical_not(ood_list))].sum()
            neg_nt_in = sum(neg_bool[np.logical_not(ood_list)])
            foil_edge_acc_in = correct_edge_in / float(neg_nt_in) * 100
            self._logger.info(
                'foil_acc_in_ora={}%={}/{}'.format(round(foil_edge_acc_in, 2), correct_edge_in, neg_nt_in))

            correct_edge_ood = (gt_foil == pred_foil)[np.logical_and(tf, ood_list)].sum()
            neg_nt_ood = sum(neg_bool[ood_list])
            foil_edge_acc_ood = correct_edge_ood / float(neg_nt_ood) * 100
            self._logger.info(
                'foil_acc_ood={}%={}/{}'.format(round(foil_edge_acc_ood, 2), correct_edge_ood, neg_nt_ood))

            correct_edge_ood = (gt_foil == pred_foil)[np.logical_and(neg_bool, ood_list)].sum()
            neg_nt_ood = sum(neg_bool[ood_list])
            foil_edge_acc_ood = correct_edge_ood / float(neg_nt_ood) * 100
            self._logger.info(
                'foil_acc_ood_ora={}%={}/{}'.format(round(foil_edge_acc_ood, 2), correct_edge_ood,
                                                    neg_nt_ood))


        if self._output_dir and self.max_acc < acc:
            self.max_acc = acc
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "best_predictions.pth")
            dump_st = time.time()
            with PathManager.open(file_path, "wb") as f:
                torch.save(final_preds, f)
            self._logger.info('saving preds take {} sec'.format(time.time() - dump_st))
            # todo: handle multiple losses

        loss = np.mean(final_preds['loss'])

        ret_preds = OrderedDict()
        ret_preds['val/retrieval_acc'] = grdonly_acc
        ret_preds['val/retr_nc'] = grd_nc
        ret_preds['val/retr_nt'] = pos_nt
        ret_preds['val/loss'] = loss

        match_loss_mean = np.mean(final_preds['match_loss'])
        ret_preds['val/match_loss'] = match_loss_mean
        ret_preds['val/match_acc'] = match_acc
        ret_preds['val/match_true_acc'] = 100 * (match_target == match_pred)[match_target == True].sum() / float(
            (match_target == True).sum())
        ret_preds['val/match_false_acc'] = 100 * (match_target == match_pred)[match_target == False].sum() / float(
            (match_target == False).sum())
        # ret_preds['val/overall_acc'] = overall_acc

        csv_write = [match_acc, grdonly_acc, foil_edge_acc]
        csv_path = os.path.join(self._output_dir, 'val.csv')
        if not os.path.exists(csv_path):
            head = ['match_acc', 'grd_acc', 'mrr_acc']
            pd.DataFrame([head]).to_csv(csv_path, mode='w', index=False, header=False)
        pd.DataFrame([csv_write]).to_csv(csv_path, mode='a', index=False, header=False)
        return ret_preds

    def _get_length_in_batch(self, sids):
        lengths = np.zeros_like(sids).reshape((-1,))
        for i in range(len(sids)):
            lengths[i] = len(self.exps[sids[i]]['expression'].split())
        return lengths.astype(int)