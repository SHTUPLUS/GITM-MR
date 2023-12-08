import torch
import torch.nn as nn
import numpy as np

import detectron2.utils.refdet_basics as basics
from detectron2.modeling.refdet_components.language import RNNEncoder, RNNEncoderForPretrain, ModuleInputAttention
from detectron2.modeling.refdet_components.modules import (AttendLocationModule, AttendNodeModule,
    AttendRelationModule, NormAttnMap, BottomUpPropagate, TopDownPropagate,
                                                           MinScorePoolingPredictor, RegressionHead)
from detectron2.modeling.losses.triplet import SoftmaxLoss

def build_refdet_head(cfg):
    return RCRN(cfg)

class RCRN(nn.Module):
    def __init__(self, cfg):

        super(RCRN, self).__init__()

        self.vis_feat_type = cfg.MODEL.REF.VIS_FEAT_TYPE

        self.vec_sim = 'sub'
        self.vec_sim_rel = False
        self.dual_atn = False
        self.ctx = 'object_ord'
        self.ctx_top_K = 5

        if self.vis_feat_type == 'uniter':
            dim_vis_feat = 768
            word_embedding_size = 768
            self.rnn_encoder = RNNEncoderForPretrain(word_embedding_size=word_embedding_size,
                                      hidden_size=cfg.MODEL.REF.RNN.HIDDEN_SIZE,
                                      bidirectional=cfg.MODEL.REF.RNN.BIDIR,
                                      input_dropout_p=cfg.MODEL.REF.RNN.WORD_DROPOUT,
                                      dropout_p=cfg.MODEL.REF.RNN.RNN_DROPOUT,
                                      n_layers=cfg.MODEL.REF.RNN.NUM_LAYERS, # todo: should add dropout before the final layer, so nlayer>1 if dropout on
                                      rnn_type=cfg.MODEL.REF.RNN.RNN_TYPE,
                                      variable_lengths=cfg.MODEL.REF.RNN.VAR_LEN,)
        else:
            dim_vis_feat = cfg.MODEL.REF.VIS_INDIM
            word_embedding_size=cfg.MODEL.REF.RNN.WORD_EMBSIZE
            self.rnn_encoder = RNNEncoder(vocab_size=cfg.MODEL.REF.NUM_VOCAB, # TODO: VOCAB SIZE
                                        word_embedding_size=word_embedding_size,
                                        hidden_size=cfg.MODEL.REF.RNN.HIDDEN_SIZE,
                                        bidirectional=cfg.MODEL.REF.RNN.BIDIR,
                                        input_dropout_p=cfg.MODEL.REF.RNN.WORD_DROPOUT,
                                        dropout_p=cfg.MODEL.REF.RNN.RNN_DROPOUT,
                                        n_layers=cfg.MODEL.REF.RNN.NUM_LAYERS, # todo: should add dropout before the final layer, so nlayer>1 if dropout on
                                        rnn_type=cfg.MODEL.REF.RNN.RNN_TYPE,
                                        variable_lengths=cfg.MODEL.REF.RNN.VAR_LEN,
                                        pretrain=True,)

        word_ctxfeat_dim = cfg.MODEL.REF.RNN.HIDDEN_SIZE * (2 if cfg.MODEL.REF.RNN.BIDIR else 1)
        self.weight_spo_num = 4
        self.match_score_top = 5
        self.weight_module_init_s = nn.Sequential(nn.Linear(word_ctxfeat_dim+self.match_score_top*2, self.weight_spo_num),
                                                    nn.Sigmoid())
        self.weight_module_init_m = nn.Sequential(nn.Linear(word_ctxfeat_dim+self.match_score_top*2, self.weight_spo_num),
                                                          nn.Sigmoid())

        self.node_input_encoder = ModuleInputAttention(word_ctxfeat_dim)
        self.relation_input_encoder = ModuleInputAttention(word_ctxfeat_dim)
        self.location_input_encoder = ModuleInputAttention(word_ctxfeat_dim)
        
        dim_vis_innorm = cfg.MODEL.REF.VIS_INNORM
        dim_joint = cfg.MODEL.REF.JOINT_DIM
        dropout_joint = cfg.MODEL.REF.JOINT_DROPOUT
        dim_wordemb = word_embedding_size

        self.node_module = AttendNodeModule(dim_vis_feat, dim_vis_innorm, dim_joint,
                                            dim_wordemb, dropout_joint, vec_sim=self.vec_sim, dual_match=self.dual_atn)
        self.location_module = AttendLocationModule(dim_vis_innorm, dim_joint,
                                                dim_wordemb, dropout_joint, vec_sim=self.vec_sim, dual_match=self.dual_atn)
        self.relation_module = AttendRelationModule(dim_vis_feat, dim_vis_innorm, dim_joint,
                                                             dim_wordemb, dropout_joint,
                                                             vec_sim=self.vec_sim_rel)
        self.norm_fun = NormAttnMap()

        self.auto_weight = cfg.MODEL.REF.AUTO_WEIGHT
        self.match_loss_weight = cfg.MODEL.REF.MATCH_LOSS_WEIGHT
        self.ret_loss_weight = cfg.MODEL.REF.RET_LOSS_WEIGHT
        self.regre_loss_weight = cfg.MODEL.REF.REGRE_LOSS_WEIGHT

        self.bottomup_propagate = BottomUpPropagate(self.ctx, 'sgmn')
        self.topdown_propagate = TopDownPropagate(self.ctx, 'sgmn')
        self.grd_bp_opt = 'log'
        self.match_bp_opt = 'log'

        self.match_scorer = MinScorePoolingPredictor(num_top=self.match_score_top, use_transformation=True,
                                                             use_norm=False)
        self.bceloss = nn.BCEWithLogitsLoss()
        self.criterion = SoftmaxLoss()

        if self.regre_loss_weight>0:
            self.regression_head=RegressionHead(word_embedding_size,dim_vis_feat,
            joint_dim=dim_joint,loc_dim=5,visual_init_norm=dim_vis_innorm,jemb_dropout=dropout_joint)

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf Add truncated normal by ruotia...
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reorgnize_and_pad(self, pack, seq_startend_idx):
        """

        :param pack: bs x nseq
        :param seq_startend_idx: bs, 2
        :return:
        """
        ent_batch = [pack[tups[0][0]:tups[0][1]] for tups in seq_startend_idx]
        spo_batch = [pack[tups[1][0]:tups[1][1]] for tups in seq_startend_idx]

        padded_ent_batch = torch.nn.utils.rnn.pad_sequence(ent_batch, batch_first=True)
        padded_spo_batch = torch.nn.utils.rnn.pad_sequence(spo_batch, batch_first=True)

        if not padded_spo_batch.sum().is_nonzero():
            padded_spo_batch = torch.zeros_like(padded_ent_batch)

        return padded_ent_batch, padded_spo_batch

    def check_anom(self, tens):
        return torch.isinf(tens).sum() > 0 or torch.isnan(tens).sum() > 0


    def forward(self, inp, device, training_mode=True):

        if training_mode: self.criterion.train()
        seq_related, word_idx_seq, arrs= [inp[k] for k in ['seq', 'widx_seq', 'arrs']]
        # encode sentence
        adj_matrix, parent_matrix, referents = seq_related
        word_idx_seq = word_idx_seq.to(device)
        batch_seq_idx = arrs[0]
        if self.vis_feat_type == 'uniter':
            pretrained_phrase_feats=inp['phrase_feat']
        else:
            pretrained_phrase_feats=None

        ret = self.rnn_encoder(word_idx_seq, pretrained_phrase_feats)  # bsxnseq, nword, dim
        # bs,padded_nseq,ntoken,dim

        (s_ctxfeat_seq,spo_ctxfeat_seq), (s_holfeat,spo_holfeat), (s_wembfeat_seq,spo_wembfeat_seq), (s_word_idx_seq,spo_word_idx_seq)\
            = [self.reorgnize_and_pad(x, batch_seq_idx) if x is not None else (None,None) for x in list(ret[:-1]) + [word_idx_seq]]

        # make sure word-idx==0 is pad token ==> this must be true since previously we use zero pads
        s_input_labels = (s_word_idx_seq!=0).float() # filter out the padded, s_word_idx_seq: bs,padnseq,ntoken
        spo_input_labels = (spo_word_idx_seq!=0).float()
        # in an attempt to learn to soft weight the phrase according to certain aspect
        ## inputs: bs,padnseq,padntoken, dim
        ## outputs: bs,padnseq,dim (different aspects aggregated features for each phrase)
        
        s_atnfeats, s_selfatn = self.node_input_encoder(s_ctxfeat_seq, s_wembfeat_seq, s_input_labels)
        sloc_atnfeats, sloc_selfatn = self.location_input_encoder(s_ctxfeat_seq, s_wembfeat_seq, s_input_labels)
        spo_atnfeats, spo_selfatn = self.relation_input_encoder(spo_ctxfeat_seq, spo_wembfeat_seq, spo_input_labels)


        ############## message initialize #################

        roi_gt, ctx_related = [inp[k] for k in ['roi_gt', 'ctx']]
        roicoord, roicls, roi_appfeat, roi_locfeat, relloc_table, \
        gtidx, gtcoord = [x.to(device) if x is not None else x for x in roi_gt]
        
        roi_ctx_idx_seq, roi_relloc_feat, roi_ctx_idx_mask = [x.to(device) for x in ctx_related]
        
        bs,num_max_box,num_ctx_box = roi_ctx_idx_seq.shape
        num_box_dummy = num_max_box
        ent_padded_nseq = s_atnfeats.shape[1]

        s_atn_ent = self.node_module(roi_appfeat, s_atnfeats, roicls)  # bs,nseq,n
        s_atn_loc = self.location_module(roi_locfeat, sloc_atnfeats, roicls)  # bs,nseq,n

        weights_spo = self.weight_module_init_s(torch.cat((s_holfeat.detach(), torch.sort(s_atn_ent.clone().detach(), descending=True)[0][...,:self.match_score_top],
                                               torch.sort(s_atn_loc.clone().detach(), descending=True)[0][...,:self.match_score_top]), -1)) # bs, num_seq, 3
        weights_spo_expand = weights_spo.unsqueeze(2).expand(bs, s_atn_ent.shape[1], num_box_dummy, self.weight_spo_num)
        entity_grd_scores = s_atn_ent * weights_spo_expand[:, :, :, 0] + s_atn_loc * weights_spo_expand[:, :, :, 1]
        entity_grd_scores, _ = self.norm_fun(entity_grd_scores)

        entity_grd_scores = (entity_grd_scores + 1)*0.5 + 1e-6

        num_max_spo = spo_atnfeats.shape[1]
        roi_ctx_idx_seq = roi_ctx_idx_seq.unsqueeze(1).repeat((1, num_max_spo, 1, 1))
        roi_ctx_idx_mask = (roicls != -1).unsqueeze(1).unsqueeze(3).repeat((1, num_max_spo, 1, 5)).long()
        for i in range(bs):
            adj_mat = adj_matrix[i]
            for j in range(num_max_spo):
                if j not in adj_mat:
                    roi_ctx_idx_seq[i, j, ...] = -1
                    roi_ctx_idx_mask[i, j, ...] = 0
                    continue
                obj_idx = np.where(adj_mat == j)[1][0]
                obj_score = entity_grd_scores[i][obj_idx]
                obj_score_idx = torch.argsort(obj_score, descending=True)[:self.ctx_top_K]
                obj_score_idx = torch.where(obj_score_idx!=num_max_box, obj_score_idx, torch.argsort(obj_score, descending=True)[5])
                roi_ctx_idx_seq[i, j, :, :] = obj_score_idx

        spo_offset_idx = torch.tensor(np.array(range(bs)) * num_max_box, requires_grad=False).to(device).long() # bs
        spo_offset_idx = spo_offset_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(bs, num_max_spo, num_max_box, num_ctx_box)  # bs, nspo, nbox, nctx
        roi_mask = roi_ctx_idx_mask # bs,nbox,nctx
        ctx_idx_adjusted =  roi_mask * roi_ctx_idx_seq
        reshape_feat = roi_appfeat.contiguous().view(bs * num_max_box, -1)
        spo_offset_idx = spo_offset_idx + ctx_idx_adjusted
        pseudo_bs = bs*num_max_spo # view as a larger batch size, for multiple spo

        roi_ctxfeat = torch.index_select(reshape_feat, 0, spo_offset_idx.reshape(-1))  # use bsxnumbox ind to index roi features
        # bs,nbox,nctx,dim
        roi_ctxfeat = roi_ctxfeat.contiguous().view(pseudo_bs, num_max_box, num_ctx_box, -1) * roi_mask.reshape(pseudo_bs, num_max_box, num_ctx_box, 1).float()


        # Online version of roi_relloc_feat
        roi_centers = torch.zeros((bs, num_max_box, 2)).float().to(device)
        roi_centers[:, :, 0] = roi_locfeat[:, :, 0] + roi_locfeat[:, :, 2]/2
        roi_centers[:, :, 1] = roi_locfeat[:, :, 1] + roi_locfeat[:, :, 3]/2
        roi_centers = roi_centers.unsqueeze(2).expand((bs, num_max_box, num_ctx_box, 2))

        reshape_loc_feat = roi_locfeat.contiguous().view(bs * num_max_box, -1)
        roi_ctx_locfeat = torch.index_select(reshape_loc_feat, 0, spo_offset_idx.reshape(-1))
        roi_ctx_locfeat = roi_ctx_locfeat.contiguous().view(pseudo_bs, num_max_box, num_ctx_box, -1) * roi_mask.reshape(pseudo_bs, num_max_box, num_ctx_box, 1).float()
        expand_locfeat = roi_locfeat.unsqueeze(2).expand((bs, num_max_box, num_ctx_box, -1))

        roi_centers = roi_centers.unsqueeze(1).expand(bs, num_max_spo, num_max_box, num_ctx_box, -1).reshape(pseudo_bs, num_max_box, num_ctx_box, -1)
        expand_locfeat = expand_locfeat.unsqueeze(1).expand(bs, num_max_spo, num_max_box, num_ctx_box, -1).reshape(pseudo_bs, num_max_box, num_ctx_box, -1)

        roi_relloc_feat_online = torch.zeros(pseudo_bs, num_max_box, num_ctx_box, 5).float().to(device)
        roi_relloc_feat_online[..., 0] = (roi_ctx_locfeat[..., 0] - roi_centers[..., 0]) / expand_locfeat[..., 2]
        roi_relloc_feat_online[..., 1] = (roi_ctx_locfeat[..., 1] - roi_centers[..., 1]) / expand_locfeat[..., 3]
        roi_relloc_feat_online[..., 2] = (roi_ctx_locfeat[..., 0] + roi_ctx_locfeat[..., 2] - roi_centers[..., 0]) / expand_locfeat[..., 2]
        roi_relloc_feat_online[..., 3] = (roi_ctx_locfeat[..., 1] + roi_ctx_locfeat[..., 3] - roi_centers[..., 1]) / expand_locfeat[..., 3]
        roi_relloc_feat_online[..., 4] = roi_ctx_locfeat[..., 4] / expand_locfeat[..., 4]
        roi_relloc_feat_online = torch.where(torch.isnan(roi_relloc_feat_online), torch.full_like(roi_relloc_feat_online, 0), roi_relloc_feat_online)
        roi_relloc_feat = roi_relloc_feat_online

        roi_subjfeat = roi_appfeat.reshape(bs, 1, num_max_box, 1, -1).expand(bs, num_max_spo, num_max_box, num_ctx_box, -1)\
                    .reshape(pseudo_bs, num_max_box, num_ctx_box, -1)

        spo_atn = self.relation_module(roi_subjfeat, roi_ctxfeat, roi_relloc_feat, spo_atnfeats)

        weights_spo_m = self.weight_module_init_m(torch.cat((s_holfeat.detach(), torch.sort(s_atn_ent.clone().detach(), descending=True)[0][...,:self.match_score_top],
                                                     torch.sort(s_atn_loc.clone().detach(), descending=True)[0][...,:self.match_score_top]), -1))# bs, num_seq, 3

        weights_spo_expand_m = weights_spo_m.unsqueeze(2).expand(bs, s_atn_ent.shape[1], num_box_dummy, self.weight_spo_num)
        entity_match_scores = s_atn_ent * weights_spo_expand[:, :, :, 0] + s_atn_loc * weights_spo_expand[:, :, :, 1]
        entity_match_scores, _ = self.norm_fun(entity_match_scores)

        entity_match_scores = (entity_match_scores + 1)*0.5 + 1e-6

        spo_atn_match = spo_atn


        ################# propagation preparation #################

        adj_matrices = adj_matrix # list of dict[node=list(node)]
        traversal_lists = []
        total_lists = [] # for getting the whole path
        total_spo_lists = []

        for i, (adjmat,ref) in enumerate(zip(adj_matrices, referents)):
            bfs_list = [ref]
            bfs_all_list = [ref]
            visited_nodes = []
            visited_all_nodes = []
            visited_spos = []
            count = 0
            while len(bfs_list)>0:
                count += 1
                if count>s_input_labels[i,:,0].sum():
                    print("A loop may exist in:", arrs[2][i])
                    break
                current = bfs_list.pop(0)
                if current not in visited_nodes: visited_nodes.insert(0, current)
                children_idxes = np.where(adjmat[current]!=-1)[0] #adjmat[current][adjmat[current]!=-1]
                visited_spos.extend([spo_id for spo_id in adjmat[current] if spo_id != -1])
                should_visit = [idx for idx in children_idxes if (adjmat[idx]!=-1).sum()!=0 ] # if the children nodes don't have next level
                should_visit_all = [idx for idx in children_idxes]
                bfs_all_list.extend(should_visit_all)
                bfs_list.extend(should_visit)

            traversal_lists.append(np.array(visited_nodes))
            while len(bfs_all_list)>0: # for getting the whole path
                visited_all_nodes.insert(0, bfs_all_list.pop(0))
            total_lists.append(np.array(visited_all_nodes))
            total_spo_lists.append(np.array(visited_spos))

        max_len_total = max([len(x) for x in total_lists])
        padded_total_lists = np.stack([np.pad(tl, (max_len_total-len(tl), 0), constant_values=-1) for tl in total_lists])
        max_len_spo = max([len(x) for x in total_spo_lists])
        padded_total_spo = np.stack([np.pad(tl, (max_len_spo-len(tl), 0), constant_values=-1) for tl in total_spo_lists])

        roicls_match = -torch.ones_like(roicls).cuda()
        roi_mask_match = torch.zeros_like(roi_mask).cuda()
        sorted_score_idxes = basics.to_numpy(torch.argsort(entity_grd_scores, dim=2, descending=True))
        for i in range(sorted_score_idxes.shape[0]):
            for j in range(len(adj_matrices[i])):
                roicls_match[i, sorted_score_idxes[i, j, :self.ctx_top_K]] = 1.0e6
                for k in adj_matrices[i][j]:
                    if k == -1: continue
                    if self.ctx == 'object_ord':
                        roi_mask_match[i, int(k), sorted_score_idxes[i, j, :self.ctx_top_K], :] = 1
                    else:
                        roi_mask_match[i, sorted_score_idxes[i, j, :self.ctx_top_K], :] = 1


        ########################### propagating ###########################

        topdown_traversal = []
        for i, (adjmat, ref) in enumerate(zip(adj_matrices, referents)):
            bfs_list = [ref]
            visited_nodes = []
            # visited_spos = []
            count = 0
            while len(bfs_list)>0:
                count += 1
                if count>s_input_labels[i,:,0].sum():
                    print("A loop may exist in:", arrs[2][i])
                    break

                current = bfs_list.pop(0)
                if current not in visited_nodes and current != ref: visited_nodes.append(current)
                children_idxes = np.where(adjmat[current]!=-1)[0] #adjmat[current][adjmat[current]!=-1]
                # visited_spos.extend([spo_id for spo_id in adjmat[current] if spo_id != -1])
                should_visit = [idx for idx in children_idxes]
                bfs_list.extend(should_visit)
            topdown_traversal.append(np.array(visited_nodes).astype(int))

        entity_grd_scores = self.bottomup_propagate(traversal_lists, adj_matrices, entity_grd_scores, spo_atn,
                                                    ctx_idx_adjusted, roicls, roi_mask, weights_spo_expand[..., 2], self.grd_bp_opt)

        entity_match_scores_bp = self.bottomup_propagate(traversal_lists, adj_matrices, entity_match_scores.clone(), spo_atn_match,
                                                          ctx_idx_adjusted, roicls_match, roi_mask_match, weights_spo_expand_m[..., 2], self.match_bp_opt)

        entity_match_scores_td = self.topdown_propagate(topdown_traversal, adj_matrices, entity_match_scores.clone(), spo_atn_match,
                                                        ctx_idx_adjusted, roicls_match, roi_mask_match, weights_spo_expand_m[..., 3], self.match_bp_opt)


        ###################### propagate finish ##########################

        offset_referents_inbatch = [idx + sampleid * ent_padded_nseq for sampleid, idx in enumerate(referents)]
        logits = torch.index_select(entity_grd_scores.reshape((-1, num_box_dummy)), 0,
                                    torch.as_tensor(offset_referents_inbatch).to(device))

        atn_track = []
        atn_mask = []
        for t in range(max_len_total):
            ent_node_idx_inbatch = padded_total_lists[:, t]
            offset_ent_idx_inbatch = [idx + sampleid * ent_padded_nseq for sampleid, idx in
                                      enumerate(ent_node_idx_inbatch)]
            if offset_ent_idx_inbatch[0] == -1:
                offset_ent_idx_inbatch[0] = 0

            if self.match_score_top != num_max_box:
                match_inp = torch.sort(entity_match_scores_bp, dim=2, descending=True)[0]
            else:
                match_inp = entity_match_scores_bp
            atn_track.append(torch.index_select(match_inp.reshape((-1, num_max_box)), 0,
                                                    torch.as_tensor(offset_ent_idx_inbatch).to(device)))
            atn_mask.append(torch.FloatTensor(ent_node_idx_inbatch != -1).to(device))

            if self.match_score_top != num_max_box:
                match_inp_td = torch.sort(entity_match_scores_td, dim=2, descending=True)[0]
            else:
                match_inp_td = entity_match_scores_td
            atn_track.append(torch.index_select(match_inp_td.reshape((-1, num_max_box)), 0,
                                            torch.as_tensor(offset_ent_idx_inbatch).to(device)))
            atn_mask.append(torch.FloatTensor(ent_node_idx_inbatch != -1).to(device))

        atn_track_stacked = torch.stack(atn_track, 0) # only for combining edges
        match_logits = self.match_scorer(atn_track_stacked, torch.stack(atn_mask, 0), roicls)['match']

        atn_logits = []
        for atn, mask in zip(atn_track, atn_mask):
            atn_logits.append(self.match_scorer(torch.unsqueeze(atn.clone(), 0), torch.unsqueeze(mask, 0), roicls)['match'])
        node_logits = torch.stack(atn_logits, 1)

        if not training_mode:
            min_pos = []
            min_logits = []
            for i in range(bs):
                adjmat = adj_matrices[i]
                min_edge_logits = 99999
                min_edge_id = 0
                # word_pos = 0
                for idx, spo_id in enumerate(padded_total_spo[i]):
                    if spo_id == -1: continue
                    parent = np.where(adjmat == spo_id)[0][0]
                    child = np.where(adjmat == spo_id)[1][0]
                    parent_id = np.where(padded_total_lists[i] == parent)[0][0]
                    child_id = np.where(padded_total_lists[i] == child)[0][0]
                    edge_logits_bp = node_logits[i][parent_id*2] - node_logits[i][child_id*2]
                    edge_logits_td = node_logits[i][child_id*2+1] - node_logits[i][parent_id*2+1]
                    edge_logits_sum = torch.add(edge_logits_bp, edge_logits_td)
                    if edge_logits_sum < min_edge_logits:
                        min_edge_logits = edge_logits_sum
                        min_edge_id = int(spo_id)
                        # word_pos = torch.argmax(spo_selfatn[i * num_max_spo + int(spo_id)]).item()
                if min_edge_logits == 99999: min_edge_logits = torch.tensor([0.]).to(device).detach()
                min_pos.append(min_edge_id)
                min_logits.append(min_edge_logits)

            min_pos = np.array(min_pos)


        ##########grd loss and prediction##########
        if training_mode: self.criterion.train()
        all_loss, score = self.criterion(logits, roicls, gtidx, match=arrs[-1])

        roicls = basics.to_numpy(roicls)
        final_score = basics.to_numpy(logits.detach())
        final_score[roicls == -1] = -999
        pred_ind = np.argmax(final_score, 1)

        roicoord = basics.to_numpy(roicoord.detach())

        regression_loss,pred_coors_regre = self.regression_head(pred_ind, s_atnfeats, roi_appfeat, roi_locfeat, proposals=roicoord, gt_coors=gtcoord, phrase_mask=s_input_labels[:,:,0],match_mask=arrs[-1])
        all_loss['regre_loss']=regression_loss*self.regre_loss_weight
        pred_coord=np.asarray(pred_coors_regre.detach().cpu(),dtype=np.float)

        sorted_ind = np.argsort(final_score, axis=1)[:, ::-1]  # reverse order from large to small score
        sorted_scores = [sample_fscore[sample_sind] for sample_fscore, sample_sind in
                         zip(final_score, sorted_ind)]  # list of nparr
        sorted_rois = [sample_rois[indseq] for indseq, sample_rois in zip(sorted_ind, roicoord)]

        gtidx = basics.to_numpy(gtidx)

        ncorrect = (gtidx == pred_ind).sum()
        ntotal = pred_ind.size
        img_ids, sent_ids, unqiue_ids, num_level = arrs[1:5]
        metrics = dict(ncorrect=ncorrect, ntotal=ntotal, pred_ind=pred_ind, pred_coord=pred_coord,
                       gt_ind=basics.to_numpy(gtidx), gt_coord=basics.to_numpy(gtcoord),
                       img_ids=img_ids, sent_ids=sent_ids, unique_ids=unqiue_ids, full_path_lengths=[len(x) for x in traversal_lists],
                       sorted_rois=sorted_rois, sorted_scores=sorted_scores)

        all_loss['loss'] = all_loss['loss'] * self.ret_loss_weight

        match_list = arrs[-1]
        metrics['counter_type'] = arrs[-2]  # if use counter, counter type is 1th
        metrics['match'] = match_list
        ncorrect = (gtidx == pred_ind)[match_list].sum()
        ntotal = sum(match_list)
        metrics['ncorrect'] = ncorrect
        metrics['ntotal'] = ntotal
        metrics['num_all'] = len(match_list)

        pred_match = basics.to_numpy(match_logits.reshape(match_list.shape) > 0)
        match_loss = self.bceloss(match_logits.reshape(match_list.shape),
                                  torch.tensor(match_list).float().to(device))


        ########################
        # gives the prediction
        all_loss['match_loss'] = match_loss
        all_loss['match_loss'] = all_loss['match_loss'] * self.match_loss_weight
        metrics['match_logits'] = basics.to_numpy(match_logits.detach())
        if not training_mode:
            metrics['pred_foil'] = min_pos
            metrics['gt_foil'] = arrs[-5]

        match_correct = (pred_match == match_list).sum()
        metrics['pred_match'] = pred_match  # 1 to be match, otherwise not match
        metrics['num_matchcorrect'] = match_correct
        true_nc = (pred_match == match_list)[match_list == True].sum()
        false_nc = (pred_match == match_list)[match_list == False].sum()
        true_nt = (match_list == True).sum()
        false_nt = (match_list == False).sum()
        metrics['true_nc'] = true_nc
        metrics['false_nc'] = false_nc
        metrics['true_nt'] = true_nt
        metrics['false_nt'] = false_nt

        return all_loss, metrics