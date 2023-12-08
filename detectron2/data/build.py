# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch.utils.data
import time
import itertools
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng

from .catalog import DatasetCatalog
from .common import DatasetFromList, MapDataset
from .dataset_mapper import RefdetMapperFromMemV2, RefdetMapperFromMemPretrain
from .samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler

import re
from torch.nn.utils.rnn import pad_sequence
"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_refdet_train_loader",
    "build_refdet_test_loader",
]

def get_refdet_dataset_dicts(dataset_names):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (list[str]): a list of dataset names

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    assert len(dataset_names)==1, 'Dataset len is {}, more than 1'.format(len(dataset_names))
    dataset_dicts, dataset_depend_args = DatasetCatalog.get(dataset_names[0])
    assert len(dataset_dicts), "Dataset '{}' is empty!".format(dataset_names[0])

    return dataset_dicts, dataset_depend_args # list of dicts, [numtoken, numbox]

def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, num_workers=0,
):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    collocator = Collocator.ours_collocation
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collocator,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=False
    )

def get_proposal_name(datasetname):
    if 'prop' in datasetname:
        return 'vinvl'
    elif 'vinvlrcnn' in datasetname:
        return 'vinvl_feat4rcnn'
    elif 'fastrcnn' in datasetname:
        return 'fastrcnn'
    else: return ''

def get_split_version_name(datasetname):
    split_name, version_name = 'train', 1
    for split in ['train', 'val', 'testA', 'testB', 'test']:
        if split in datasetname:
            split_name = split
            break
    if 'v' in datasetname:
        res = re.search('v\d{1,}', datasetname)
        if res is not None:
            version_name = int(res.group()[1:])

    return split_name, version_name

# uniter data tools
def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size

    # outsize is the max length of origin txt len+box num in this batch
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index


def build_refdet_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.

    Returns:
        an infinite iterator of training data
    """
    st = time.time()
    dataset_dicts, dataset_depend_args = get_refdet_dataset_dicts(
        cfg.DATASETS.TRAIN,
    )
    print('dataset dicts loading takes {} secs.'.format(round(time.time()-st),2))

    dataset = DatasetFromList(dataset_dicts, copy=False) # to torch dataset
    use_counter = 'counter' in cfg.DATASETS.TRAIN[0]

    split, _ = get_split_version_name(cfg.DATASETS.TRAIN[0])

    if mapper is None:
        print('mapper in build train')
        
        if cfg.MODEL.REF.VIS_FEAT_TYPE == 'uniter':
            mapper = RefdetMapperFromMemPretrain(cfg.DATAROOT, dataset_depend_args, split=split, max_nprop=cfg.DATASETS.MAX_PROP_NUM)
        else:
            mapper = RefdetMapperFromMemV2(cfg.DATAROOT, dataset_depend_args, split=split, max_nprop=cfg.DATASETS.MAX_PROP_NUM)

    dataset = MapDataset(dataset, mapper) # add mapper

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if use_counter:
        logger.info('negative sampler stats:')
        logger.info(dataset_depend_args['levels'])

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    logger.info('start building dataloader')
    
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    ),len(dataset)

def build_refdet_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts, dataset_depend_args = get_refdet_dataset_dicts(
        [dataset_name],
    )
    dataset = DatasetFromList(dataset_dicts)
    split, version = get_split_version_name(cfg.DATASETS.TEST[0])

    if mapper is None:
        print('mapper in build test')

        if cfg.MODEL.REF.VIS_FEAT_TYPE == 'uniter':
            mapper = RefdetMapperFromMemPretrain(cfg.DATAROOT, dataset_depend_args, split=split, max_nprop=cfg.DATASETS.MAX_PROP_NUM)
        else:
            mapper = RefdetMapperFromMemV2(cfg.DATAROOT, dataset_depend_args, split=split, max_nprop=cfg.DATASETS.MAX_PROP_NUM)

    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.

    world_size = get_world_size()
    total_batch_size = cfg.SOLVER.IMS_PER_BATCH_TEST
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    collocator = Collocator.ours_collocation
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collocator,
        pin_memory=False
    )

class Collocator:
    @classmethod
    def before_collocation(cls, batch):
        
        num_items_each_example = len(batch[0])
        result = [[example[item] for example in batch] for item in range(num_items_each_example)]
        
        return result

    @classmethod
    def get_fixed_part(cls, result):
        # roi_related and gt_related is fixed
        tmp = [torch.as_tensor(np.array(x)) if x[0] is not None else None for x in result[:7]] # roi + gt
        if result[-1][0] is None:
            images = None
        else:
            images = result[-1]

        return dict(roi_gt=tmp, images=images)

    @classmethod
    def default_collocation(self, batch):
        result = self.before_collocation(batch)
        d = self.get_fixed_part(result)
        return result, d

    @classmethod
    def uniter_collocation(cls, result):
        (input_ids, img_feats, img_pos_feats, attn_masks, attn_masks_txt, attn_masks_img, obj_masks,
         token_split_num_list, entity_spo_positions, entity_positions) = result[-11:-1]

        txt_lens = [i.size(0) for i in input_ids]
        num_bbs = [f.size(0) for f in img_feats]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
        if attn_masks[0] is not None:
            attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
            out_size = attn_masks.size(1)
            bs, max_tl = input_ids.size()
            gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
        else:
            attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
            attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)
            bs, max_tl = input_ids.size()
            gather_index = None

        phrasenum_idx = [[[i]] * len(entity_spo_positions[i]) for i in range(bs)]
        phrasenum_idx = [i for p in phrasenum_idx for i in p]
        entity_spo_positions = [torch.tensor(i) for p in entity_spo_positions for i in p]
        entity_spo_positions = pad_sequence(entity_spo_positions, batch_first=True, padding_value=-1)

        obj_masks = pad_sequence(
            obj_masks, batch_first=True, padding_value=1)

        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'attn_masks_txt': attn_masks_txt,
            'attn_masks_img': attn_masks_img,
            'gather_index': gather_index,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs,
            'obj_masks': obj_masks,
            'phrasenum_idx': phrasenum_idx,
            'entity_spo_positions': entity_spo_positions,
            'token_split_num_list': token_split_num_list,
            'entity_positions': entity_positions
        }

    @classmethod
    def ours_collocation(self, batch):
        result, d = self.default_collocation(batch)
        d['ctx'] = [torch.as_tensor(np.asarray(x)) for x in result[7:10]]

        # todo: refine here, use number is not good
        if result[-1][0] == 'pretrained':
            arrs = [np.asarray(x) for x in result[15:-13]]
            phrase_feats = result[-12]

            if phrase_feats[0] is not None:
                phrase_feats_concat = list(itertools.chain(*phrase_feats))
                padded_phrase_feats = torch.nn.utils.rnn.pad_sequence(phrase_feats_concat, batch_first=True)
            else:
                padded_phrase_feats = None
                d.update(self.uniter_collocation(result))
            sent_word_idxs = None
        else:
            arrs = [np.asarray(x) for x in result[15:-2]]
            padded_phrase_feats = None
            # pad sentence idxs
            sent_word_idxs = result[-1]
            max_sent_len = max([len(swi) for swi in sent_word_idxs])
            sent_word_idxs = [np.pad(swi, (0, max_sent_len - len(swi))) for swi in sent_word_idxs]

        ## pack word-idx-seq: list of nseq,ntoken
        widx_seq_idx = 10
        batch_base_idx = []
        idx = 0
        max_num_token = 0
        # merge entity and spo lists in a batch, so need the start idx and end idx for each sample
        for seq in result[widx_seq_idx]:  # nseq, ntoken
            batch_base_idx.append(idx)
            idx += len(seq)
            max_num_token = max(max_num_token, seq.shape[-1])
        # batch_base_idx.append(idx)
        word_idx_seq, startend_idx, adj_matrix, parent_matrix, referent = result[widx_seq_idx:15]

        concat_list = list(
            itertools.chain(*[[np.pad(x, (0, max_num_token - len(x))) for x in seq] for seq in word_idx_seq]))
        word_idx_seq = torch.as_tensor(np.stack(concat_list))  # sum_nseq, max_num_token

        startend_idx = [((base, base + st), (base + st, base + ed)) for base, (st, ed) in
                        zip(batch_base_idx, startend_idx)]  # list

        d['seq'] = [adj_matrix, parent_matrix, referent]
        d['widx_seq'] = word_idx_seq
        d['arrs'] = [startend_idx] + arrs
        d['phrase_feat'] = padded_phrase_feats
        d['sent_word_idx'] = sent_word_idxs

        # ctx-gt, seq, widx-seq, arrs
        return d

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)