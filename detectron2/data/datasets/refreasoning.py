import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import refdet_basics as basics

import pickle as pkl
import os.path as osp
import json
import io
import time
from tqdm import tqdm
import copy
__all__ = ["register_counter_refreasoning"]


def load_counter_instances(data_root, folder, split, pad_at_first=True, debug=False, sample_method='foil_rel_len16'):
    vocab_st = time.time()
    # vocab
    dict_file = osp.join(data_root, 'word_embedding', 'vocabulary_72700.txt')
    with io.open(dict_file, encoding='utf-8') as f:
        words_idx_in_sent = [w.strip() for w in f.readlines()]
    if pad_at_first and words_idx_in_sent[0] != '<pad>':
        raise Exception("The first word needs to be <pad> in the word list.")
    word2idx = {words_idx_in_sent[n]: n for n in range(len(words_idx_in_sent))}
    print('vocab time {}'.format(round(time.time()-vocab_st, 2)))

    # refs
    # (process sample method)
    small_ids_folder = osp.join(data_root, 'small')
    if 'len' in sample_method:
        assert split == 'train'
        max_len = sample_method[-2:]
        sample_method = sample_method[:-6]
        pos_list = json.load(open(osp.join(small_ids_folder, f'{split}_small_sent_ids_{max_len}.json')))['pos']

    refs_st = time.time()
    refdb = Refer(data_root, folder, split)  # ../../data, expression, val
    refs = refdb.data['refs']
    if 'foil' in sample_method:
        refs.update(json.load(open(osp.join(data_root, folder, f'{split}_expressions_{sample_method}.json'), 'r')))
    print('refs time {}'.format(round(time.time()-refs_st, 2)))
    rois_st = time.time()
    obj_folder = 'vinvl_objects'

    with open(osp.join(data_root, obj_folder, 'rois_info.pkl'), 'rb') as f:
        rois = pkl.load(f)
    print('rois time {}'.format(round(time.time()-rois_st, 2)))

    ##### load counter mapping
    with open(osp.join(data_root, 'counter','{}_id2id_{}.pkl'.format(split, sample_method)), 'rb') as f:
        counter_examples = pkl.load(f)
    ########################
    
    max_num_box = 126
    max_sent_len = 0
    max_nseq = 0
    match_cnt,nmatch_cnt = 0,0
    dicts = []
    levels = dict()

    iter_st = time.time()
    for sentid in tqdm(refs): # num_nodes is available only for val split
        if sentid not in counter_examples: continue
        if 'len' in sample_method and sentid not in pos_list: continue

        # originally: refid, imgid, expr, bbox
        ref_struct = refs[sentid] # added by foil: foil_pos, foil_word, target_word
        ref_struct['pos_sent_id'] = sentid
        if 'foil' in sample_method:
            ref_struct['foil_pos'] = -1
            ref_struct['foil_indices'] = [-1, -1]
            ref_struct['foil_phrase'] = 'none'
            ref_struct['target_phrase'] = 'none'
        roi_related = dict()
        imgid = ref_struct['image_id']
        roi = rois[imgid]
        boxes = roi['box']
        x,y,dx,dy = ref_struct['bbox']
        x1 = x+dx -1
        y1 = y+dy -1
        iou, gtidx, ious = basics.max_overlap([x,y,x1,y1], boxes)
        ref_struct['gt_idx'] = gtidx
        ref_struct['gt_xyxy'] = np.array([x,y,x1,y1], dtype=np.float32)
        ref_struct['roi'] = roi
        # loc feat
        iw, ih = roi['size']

        locfeat = []
        for x, y, x1, y1 in boxes:
            w = x1 - x + 1
            h = y1 - y + 1
            locfeat.append([x/iw, y/ih, w/iw, h/ih, w*h/(iw*ih)])

        locfeat = np.asarray(locfeat, dtype=np.float32)
        ref_struct['locfeat'] = locfeat

        for k in ['image_id','bbox', 'gt_idx', 'gt_xyxy', 'roi', 'locfeat']:
            roi_related[k] = ref_struct[k]
        #######################################
        ref_struct['sent_id'] = sentid

        ref_struct['token'] = tokens = ref_struct['expression'].split(' ')
        max_sent_len = max(max_sent_len, len(tokens))
        # word idx
        word_idxes = [word2idx.get(word.lower(), word2idx['<unk>']) for word in tokens]
        ref_struct['word_idxes'] = word_idxes
        ref_struct['match'] = True
        ref_struct['counter_type'] = 'None'
        ref_struct['unique_id'] = match_cnt + nmatch_cnt
        match_cnt += 1
        dicts.append(ref_struct)

        ################# counter sent
        if counter_examples.get(sentid, None) is not None:
            roi_related = dict()
            ref_struct = refs[sentid]
            ref_struct['pos_sent_id'] = sentid
            
            imgid = ref_struct['image_id']
            roi = rois[imgid]
            boxes = roi['box']
            x,y,dx,dy = ref_struct['bbox']
            x1 = x+dx -1
            y1 = y+dy -1
            iou, gtidx, ious = basics.max_overlap([x,y,x1,y1], boxes)
            ref_struct['gt_idx'] = gtidx
            ref_struct['gt_xyxy'] = np.array([x,y,x1,y1], dtype=np.float32)
            ref_struct['roi'] = roi
            # loc feat
            iw, ih = roi['size']

            locfeat = []
            
            for x, y, x1, y1 in boxes:
                w = x1 - x + 1
                h = y1 - y + 1
                locfeat.append([x/iw, y/ih, w/iw, h/ih, w*h/(iw*ih)])
                
            locfeat = np.asarray(locfeat, dtype=np.float32)
            ref_struct['locfeat'] = locfeat

            for k in ['image_id','bbox', 'gt_idx', 'gt_xyxy', 'roi', 'locfeat']:
                roi_related[k] = ref_struct[k]

            counter_exp = counter_examples[sentid]
            sid = counter_exp['sentid']
            ref_struct = copy.deepcopy(refs[sid])
            ref_struct['sent_id'] = sid
            ref_struct['token'] = tokens = ref_struct['expression'].split(' ')
            max_sent_len = max(max_sent_len, len(tokens))

            # word idx
            word_idxes = [word2idx.get(word.lower(), word2idx['<unk>']) for word in tokens]
            ref_struct['word_idxes'] = word_idxes
            ref_struct['match'] = False
            ref_struct['counter_type'] = counter_exp['counter_type']
            coarse_type = counter_exp['counter_type']
            if coarse_type not in levels: levels[coarse_type] = 0
            levels[coarse_type] += 1
            ref_struct['unique_id'] = match_cnt+nmatch_cnt
            ref_struct.update(roi_related)
            dicts.append(ref_struct)
            nmatch_cnt += 1
        #########################################

        if match_cnt==1: print('iter time {}'.format(round(time.time()-iter_st, 2)))
        if debug and match_cnt > 1000: break

    print('match vs nmatch: {}/{}'.format(match_cnt, nmatch_cnt))
    print('total: {}'.format(len(dicts)))

    max_info = dict(max_nword=max_sent_len, max_nbox=max_num_box, levels=levels)
    max_info['max_nseq'] = max_nseq

    return dicts,max_info # list of dicts

REFR_CATES = [
    {"color": np.asarray([220, 20, 60])/255., "isthing": 1, "id": 1, "name": "gt"},
    {"color": np.asarray([11, 11, 220])/255., "isthing": 1, "id": 2, "name": "pred"},
]

def register_counter_refreasoning(setname, data_root, anno_folder, img_folder, split,
                                  debug=False, sample_method='foil_rel_len16'):

    DatasetCatalog.register(setname, lambda: load_counter_instances(data_root, anno_folder, split,
                                                                    debug=debug, sample_method=sample_method))
    MetadataCatalog.get(setname).set(
        img_root=osp.join(data_root, img_folder),
        split=split,
        evaluator_type='refreasoning',
        thing_classes=[x['name'] for x in REFR_CATES],
        thing_colors=[x['color'] for x in REFR_CATES], # gt, pred
    )


class Refer(object):
    # datadir: data/datasetname
    # reffile: datadir/split_expressions.json
    # sg: datadir/split_sgs.json or split_sg_seqs.json

    # data: dict
    ## dataset: name
    ## refs: dict(sentenceid=dict()
    ### expression, token, sent
    def __init__(self, data_root, dataset, split):
        print(("Loading dataset %s %s into memory..." % (dataset, split)))
        self.data_dir = osp.join(data_root, dataset)

        ref_file = osp.join(self.data_dir, f'{split}_expressions_ori_rel.json')

        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = self._load_data(ref_file)
        # self.add_token()
        print(('number of refs:', len(self.data['refs'])))

    def _load_data(self, ref_file):
        return json.load(open(ref_file, 'r'))

    def get_sentIds(self, img_ids=None):
        if img_ids is None:
            return list(self.data['refs'].keys())
        else:
            sent_ids = []
            for sent_id in sent_ids:
                if self.data['refs'][sent_id]['image_id'] in img_ids:
                    sent_ids.append(sent_id)
            return sent_ids

    def get_imgIds(self, sent_ids):
        img_ids = []
        for sent_id in sent_ids:
            img_ids.append(self.data['refs'][sent_id]['image_id'])
        img_ids = set(img_ids)
        img_ids = list(img_ids)
        return img_ids

    def load_sent(self, sent_id):
        return self.data['refs'][sent_id]

    def add_token(self):
        for sent_id in self.data['refs']:
            self.data['refs'][sent_id]['token'] = self.data['refs'][sent_id]['expression'].split(' ')
            self.data['refs'][sent_id]['sent'] = self.data['refs'][sent_id]['expression']