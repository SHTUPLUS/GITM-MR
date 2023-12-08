# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from . import detection_utils as utils
from . import transforms as T
from glob import glob
import h5py
import os.path as osp
import json
import cv2

from transformers import BertTokenizer

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["RefdetMapperFromMemV2", "RefdetMapperFromMemPretrain"]

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
import time
import pickle as pkl


class RefdetMapperFromMemV2:
    def __init__(
            self,
            data_root: str = 'data',
            dataset_depend_args: dict = None,
            split: str = 'train',
            max_nprop: int=126,
    ):
        """
        NOTE: this interface is experimental.

        Args:

        """
        #################### todo: max info?
        self.max_nprop = max_nprop
        self.max_nbox = dataset_depend_args['max_nbox']
        #########################################

        self.logger = logging.getLogger(__name__)

        obj_folder = f'vinvl_objects'
        print('obj_folder:', obj_folder)
        num_files = len(glob(osp.join(data_root, obj_folder, 'gt_objects_*.h5')))
        self.h5_paths = [osp.join(data_root, obj_folder, 'gt_objects_%d.h5' % n) for n in range(num_files)]
        self.h5files = None

        self.parse_path = osp.join(data_root, 'parse', f'{split}_graphs.pkl')
        with open(self.parse_path, 'rb') as f:
            self.parsed_structs = pkl.load(f)

        self.foil_dir = osp.join(data_root, 'parse')
        foil_path = osp.join(self.foil_dir, f'{split}_foil_rel_graphs.pkl')
        with open(foil_path, 'rb') as f:
            self.parsed_structs.update(pkl.load(f))

        print(split, self.parse_path)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        ########################################
        max_numbox = self.max_nbox
        max_numprop= self.max_nprop
        ########################################

        if self.h5files is None:
            st = time.time()
            self.h5files = [h5py.File(path, 'r', libver='latest', swmr=True)['features'] for path in self.h5_paths]
            self.logger.info('loading h5 takes {} secs'.format(round(time.time() - st, 2)))

        roi = dataset_dict['roi']
        num_box, box_coord, box_cat, file, h5idx = [roi[k] for k in ['num_objs', 'box', 'cls', 'file', 'idx']]
        roi_appfeat = self.h5files[file][h5idx]
        locfeat = dataset_dict['locfeat'][:max_numprop]

        if num_box>max_numprop:
            box_coord=box_coord[:max_numprop]
            box_cat=box_cat[:max_numprop]
            num_box=max_numprop
            roi_appfeat[max_numprop:]=0.0

        padded_boxcoord = np.pad(box_coord, ((0, max_numbox - num_box), (0, 0)), mode='constant',
                                 constant_values=0.0)  # check float32
        padded_boxcat = np.pad(box_cat, (0, max_numbox - num_box), mode='constant',
                               constant_values=-1.0)  # check float32

        padded_roifeat = roi_appfeat[:max_numbox, :].astype(np.float32)
        padded_relloc_table = None

        padded_locfeat = np.pad(locfeat, ((0, max_numbox - num_box), (0, 0)), mode='constant', constant_values=0.0)
        roi_related = [padded_boxcoord, padded_boxcat, padded_roifeat, padded_locfeat, padded_relloc_table]

        gt_idx, gt_xyxy = [dataset_dict[k] for k in ['gt_idx', 'gt_xyxy']]
        gt_related = [gt_idx, gt_xyxy]

        num_nodes = dataset_dict.get('num_nodes', None)
        meta = [dataset_dict['image_id'], dataset_dict['sent_id'], dataset_dict['unique_id'], num_nodes]

        word_idxes = dataset_dict['word_idxes']

        image = [None]

        ###############################
        padded_ctx_locfeat=np.zeros([max_numbox,5,5],dtype='float32')
        padded_ctx_idx=np.zeros([max_numbox,5],dtype='int64')
        padded_ctx_idx_mask=np.zeros([max_numbox,5],dtype='int64')

        ctx_related = [padded_ctx_idx, padded_ctx_locfeat, padded_ctx_idx_mask]
        ###############################

        sentid = dataset_dict['sent_id']
        if type(sentid) == int:
            sentid = str(sentid)
        parsed_struct = self.parsed_structs[sentid]

        entity_phrases, spo_phrases, adj_matrix, parent_matrix, referent = [parsed_struct[k]
                                                                            for k in
                                                                            ['entity_phrase_list', 'spo_phrase',
                                                                             'adj_matrix', 'parent_matrix', 'referent']]
        num_entity = len(entity_phrases)
        num_spo = len(spo_phrases)
        start_end_idx = (num_entity, num_spo + num_entity)
        all_phrases = entity_phrases + spo_phrases
        max_num_token = max([len(x) for x in all_phrases])
        phrases_as_whole = np.stack([np.pad(x, (0, max_num_token - len(x)), mode='constant', constant_values=0)
                                     for x in all_phrases])
        seq_related = [phrases_as_whole, start_end_idx, adj_matrix, parent_matrix, referent]

        last = meta + [dataset_dict['foil_pos'], dataset_dict['foil_phrase'], dataset_dict['target_phrase']]
        last = last + [dataset_dict['counter_type'], dataset_dict['match']]

        return roi_related + gt_related + ctx_related  + seq_related + last + image + [word_idxes]


    def wids_to_seq_idx(self, wids, sent_seq):
        # a naive search version
        indices = set()
        for wid in wids:
            for i, token in enumerate(sent_seq):
                if wid == token:
                    indices.add(i)
        return list(indices)


class RefdetMapperFromMemPretrain:
    def __init__(
            self,
            data_root: str = 'data',
            dataset_depend_args: dict = None,
            split: str = 'train',
            max_nprop : int= 126,
    ):
        #################### todo: max info?
        self.max_nprop = max_nprop
        self.max_nbox = dataset_depend_args['max_nbox']
        #########################################
        self.logger = logging.getLogger(__name__)

        obj_folder = 'vinvl_objects'
        print('obj_folder:', obj_folder)
        # ori image feats path
        num_files_ori = len(glob(osp.join(data_root, obj_folder, 'gt_objects_*.h5')))
        self.h5_paths_ori = [osp.join(data_root, obj_folder, 'gt_objects_%d.h5' % n)
                         for n in range(num_files_ori)]
        self.h5files_ori = None

        self.parse_path = osp.join(data_root, 'parse', f'{split}_graphs.pkl')
        with open(self.parse_path, 'rb') as f:
            self.parsed_structs = pkl.load(f)

        self.foil_dir = osp.join(data_root, 'parse')
        with open(osp.join(self.foil_dir, f'{split}_foil_rel_graphs.pkl'), 'rb') as f:
            self.parsed_structs.update(pkl.load(f))
            print(split, self.parse_path)

        with open(osp.join(data_root, 'word2token', 'positive_positions_{}.json'.format(split)),'r') as f:
            self.positive_positions = json.load(f)
        with open(osp.join(data_root, 'word2token', 'negative_positions_{}.json'.format(split)),'r') as f:
            self.negative_positions = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained("data/uniter/tools/bert-base-cased", use_fast=True)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        #####################
        max_numbox = self.max_nbox
        max_numprop= self.max_nprop
        ########################################

        roi = dataset_dict['roi']
        num_box, box_coord, box_cat, file_rel, h5idx_rel= [roi[k] for k in ['num_objs', 'box', 'cls', 'file', 'idx']]

        if self.h5files_ori is None:
            st = time.time()
            self.h5files_ori = [h5py.File(path, 'r', libver='latest', swmr=True)['features'].astype('float32') for path in self.h5_paths_ori]
            self.logger.info('loading h5 takes {} secs'.format(round(time.time() - st, 2)))

        if num_box>max_numprop:
            box_coord=box_coord[:max_numprop]
            box_cat=box_cat[:max_numprop]
            num_box=max_numprop

        padded_boxcoord = np.pad(box_coord, ((0, max_numbox - num_box), (0, 0)), mode='constant',
                                 constant_values=0.0)  # check float32
        padded_boxcat = np.pad(box_cat, (0, max_numbox - num_box), mode='constant',
                               constant_values=-1.0)  # check float32

        padded_roifeat = None
        txtfeat = None
        padded_relloc_table = None

        # loc feat
        locfeat = dataset_dict['locfeat'][:max_numprop]
        padded_locfeat = np.pad(locfeat, ((0, max_numbox - num_box), (0, 0)), mode='constant', constant_values=0.0)
        roi_related = [padded_boxcoord, padded_boxcat, padded_roifeat, padded_locfeat, padded_relloc_table]

        num_nodes = dataset_dict.get('num_nodes', None)
        gt_idx, gt_xyxy = [dataset_dict[k] for k in ['gt_idx','gt_xyxy']]

        gt_related = [gt_idx, gt_xyxy]
        meta = [dataset_dict['image_id'], dataset_dict['sent_id'], dataset_dict['unique_id'], num_nodes]

        image = [None]

        padded_ctx_locfeat=np.zeros([max_numbox,5,5],dtype='float32')
        padded_ctx_idx=np.zeros([max_numbox,5],dtype='int64')
        padded_ctx_idx_mask=np.zeros([max_numbox,5],dtype='int64')

        ctx_related = [padded_ctx_idx, padded_ctx_locfeat, padded_ctx_idx_mask]

        #####################
        sentid = dataset_dict['sent_id']
        if type(sentid) == int:
            sentid = str(sentid)
        parsed_struct = self.parsed_structs[sentid]

        entity_phrases, spo_phrases, adj_matrix, parent_matrix, referent = [parsed_struct[k]
                                                                            for k in ['entity_phrase_list', 'spo_phrase',
                                                                                      'adj_matrix','parent_matrix', 'referent']]
        num_entity = len(entity_phrases)
        num_spo = len(spo_phrases)
        start_end_idx = (num_entity, num_spo+num_entity)
        all_phrases = entity_phrases+spo_phrases
        max_num_token = max([len(x) for x in all_phrases])
        phrases_as_whole = np.stack([np.pad(x, (0, max_num_token-len(x)), mode='constant', constant_values=0)
                                     for x in all_phrases])
        seq_related = [phrases_as_whole, start_end_idx, adj_matrix, parent_matrix, referent]

        all_phrases_feat, entity_spo_positions, entity_position=self.wids_to_wordprefeats(txtfeat, sentid)

        last = meta + [dataset_dict['foil_pos'], dataset_dict['foil_phrase'], dataset_dict['target_phrase']]
        last = last + [dataset_dict['counter_type'], dataset_dict['match']]

        uniter_related=self.get_uniter_related(dataset_dict)
        uniter_related.append(entity_spo_positions)
        uniter_related.append(entity_position)

        # 5 + 2 + 3 + 5 + 9 + 1 + 1 + 10 + 1
        return roi_related + gt_related + ctx_related  + seq_related + last + image + [all_phrases_feat] + uniter_related + ['pretrained']


    # convert sentence encoded to ids to features
    def wids_to_wordprefeats_approximate(self, wids, sent_enc, sent_feat, sent_id):

        '''
        input: wids in one sentence
        output: words' pretrained features
        '''
        widpos=[]
        if sent_id=='1177268':
            for wid in wids:
                if wid not in sent_enc:
                    widpos.append(0)
                else:
                    widpos.append(sent_enc.index(wid))
            return torch.from_numpy(sent_feat[widpos])
        for wid in wids:
            if wid not in sent_enc:
                print(sent_id)
            widpos.append(sent_enc.index(wid))
        return torch.from_numpy(sent_feat[widpos])

    def wids_to_wordprefeats(self, sent_feat, sent_id):
        '''
        input: wids in one sentence
        output: words' pretrained features
        '''

        if sent_id in self.negative_positions:
            positions=self.negative_positions[sent_id]
        else:
            positions=self.positive_positions[sent_id]
        entity_position=positions['entity_position']
        entity_spo_positions=positions['entity_position']+positions['spo_position']
        if sent_feat is None:
            return None,entity_spo_positions,entity_position

        all_sent_feats=[]
        
        for posit in entity_spo_positions:
            all_sent_feats.append(torch.from_numpy(sent_feat[posit]))
        return all_sent_feats,entity_spo_positions,entity_position

    def get_uniter_related(self, dataset_dict):

        input_ids_list,token_split_num_list = self.bert_tokenizer_maintain_split_num(dataset_dict['expression'])
        input_ids = torch.tensor(input_ids_list)# no need padding

        roi=dataset_dict['roi']
        iw, ih = roi['size']
        file,idx,num_bb,boxes=[roi[k] for k in ['file','idx','num_objs','box']]
        if num_bb>self.max_nprop:
            num_bb=self.max_nprop
            boxes=boxes[:self.max_nprop]
        img_feat=torch.from_numpy(self.h5files_ori[file][idx][:num_bb])

        img_pos_feat=[]
        for x, y, x1, y1 in boxes:
            w = x1 - x + 1
            h = y1 - y + 1
            img_pos_feat.append([x/iw, y/ih, x1/iw, y1/ih, w/iw, h/ih, w*h/(iw*ih)])
        img_pos_feat = torch.tensor(img_pos_feat, dtype=torch.float32)
        
        obj_masks = torch.tensor([0]*num_bb, dtype=torch.bool)

        assert len(img_feat)==num_bb

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        attn_masks_txt = None
        attn_masks_img = None

        return [input_ids,img_feat,img_pos_feat,attn_masks,attn_masks_txt,attn_masks_img,obj_masks,token_split_num_list]
    
    def bert_tokenizer_maintain_unknown(self,sentence):
        #the transformer tokenizer will split the unknown words, here we maintain them
        if isinstance(sentence,str):
            tokens=sentence.split()
        elif isinstance(sentence,list):
            tokens=sentence
        token_ids=self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids_list=[101]+token_ids+[102]

        return input_ids_list
    
    def bert_tokenizer_maintain_split_num(self, sentence):
        # record the split words and avg their split embs after compute embs 
        split_list=[] #CLS

        tokens=sentence.split()
        
        for t in tokens:
            bert_tokens = self.tokenizer.tokenize(t) #'transformer'-->['transform','##er']
            split_list.append(len(bert_tokens))
        input_ids_list = self.tokenizer(sentence)['input_ids']

        return input_ids_list, split_list