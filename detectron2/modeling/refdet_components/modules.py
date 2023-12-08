import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from detectron2.utils.refdet_basics import NormalizeScale
from detectron2.modeling.box_regression import Box2BoxTransform
from fvcore.nn import smooth_l1_loss

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = (torch.pow(X, 2).sum(dim=dim, keepdim=True) + eps).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MinScorePoolingPredictor(nn.Module):
    def __init__(self, num_top=126, init_value=0., use_transformation=False, use_norm=False):
        super(MinScorePoolingPredictor, self).__init__()
        self.num_top = num_top
        self.pad_value = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)
        self.hidden_dim = 64
        self.belief_fc = nn.Sequential(nn.Linear(num_top, self.hidden_dim),
                                       nn.BatchNorm1d(self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.2)
                                       ) \
            if use_transformation else None
        if use_transformation: fc_indim = self.hidden_dim
        else: fc_indim = num_top

        self.match_fc = nn.Sequential(
            nn.Linear(fc_indim, 1),
            # ScaleLayer(1., 0.)
            )
        self.use_norm = use_norm
        self.use_trans = use_transformation
        self._init_weights()

    def _init_weights(self):
        if self.use_trans:
            nn.init.xavier_normal_(self.belief_fc[0].weight.data)
            nn.init.zeros_(self.belief_fc[0].bias.data)

        nn.init.xavier_normal_(self.match_fc[0].weight.data)
        nn.init.zeros_(self.match_fc[0].bias.data)

    def forward(self, all_scores, score_masks, cls, no_pooling=False):

        # - sorting TODO: do this here can reduce code complexity!
        ########### use belief
        # if self.use_belief:
        #     all_scores = self.sigmoid(all_scores)
        ##########################################
        broadcast_cls = cls.clone().unsqueeze(0).repeat(all_scores.shape[0], 1, 1)  # make it only 0/1
        all_scores[broadcast_cls == -1] = -999
        all_scores = all_scores.sort(dim=-1, descending=True)[0]
        if self.use_norm:
            cls_mask = broadcast_cls==-1
            score_mask = score_masks==0.
            norm = torch.abs(all_scores * cls_mask * score_mask).max(-1,keepdim=True)[0] # t,bs,1
            norm = torch.clamp_min(norm, 1.).detach()
            all_scores = all_scores / norm

        all_scores[broadcast_cls == -1] = self.pad_value
        score_masks = torch.as_tensor(score_masks).unsqueeze(-1).repeat(1,1,all_scores.shape[-1]).cuda() # t,bs,nbox
        all_scores[score_masks == 0] = self.pad_value
        ###########################
        if self.belief_fc is not None:
            t,bs,_ = all_scores.shape
            out = self.belief_fc(all_scores[:,:,:self.num_top].reshape((t*bs,-1)))
            in_feat = out.reshape((t,bs,-1))
        else: in_feat = all_scores[:,:,:self.num_top]
        #############################
        all_scores = self.match_fc(in_feat)  # bs,1 indexed by bs,
        all_scores[score_masks[:,:,0] == 0] = 1e6
        if no_pooling:
            scores = all_scores
        else:
            scores = torch.min(all_scores, dim=0)[0]

        ret = dict()
        ret['match'] = scores
        return ret


class BottomUpPropagate(nn.Module):
    def __init__(self, ctx, bp, tracker=False, save=None):
        super(BottomUpPropagate, self).__init__()
        self.ctx = ctx
        self.bp = bp
        self.is_sgmn = bp == 'sgmn'
        self.tracker = tracker
        self.norm_fun = NormAttnMap()
        self.save = save

    def forward(self, traversal_lists, adj_matrices, ent_attn, spo_attn, ctx_idx_adjusted, roi_cls, roi_mask, weight_on_children=None,
                opt='add'):
        
        bs = ent_attn.shape[0]
        ent_padded_nseq = ent_attn.shape[1]
        spo_padded_nseq = spo_attn.shape[1]
        num_box = ent_attn.shape[2]
        num_ctx_box = spo_attn.shape[-1]
        
        max_len = max([len(x) for x in traversal_lists])
        padded_traversal_lists = np.stack([np.pad(tl, (max_len-len(tl), 0), constant_values=-1) for tl in traversal_lists])

        if self.tracker:
            ent_tracker = [[] for _ in range(bs)]
            child_tracker = [[] for _ in range(bs)]
            edge_tracker = [[] for _ in range(bs)]

        for it in range(max_len):
            if self.bp == "none":
                break
            parents = padded_traversal_lists[:,it] # bs, : node index in each sample
            offset_parents_inbatch = [idx+sampleidx*ent_padded_nseq if idx!=-1 else -1 for sampleidx,idx in enumerate(parents)]
            children_idxes_offseted_inbatch = []
            children_belongs_to_which_example = []
            edges_inbatch = []
            offset_edges_inbatch = []
            nchildrens = []
            no_parent_or_no_children = []
            bs_mask = []
            for sampleidx in range(len(parents)): # for each sample in the batch
                adjmat = adj_matrices[sampleidx]
                parent = parents[sampleidx] # we are sure that all parents must have children
                tmp = [idx for idx in range(adjmat.shape[1]) if adjmat[parent,idx]!=-1]

                if parent==-1 or len(tmp)==0: # when there is only root, it has no children
                    children = [0]
                    no_parent_or_no_children.append(0)
                    edges = [0]
                    bs_mask.append(False)
                else:
                    children = tmp
                    no_parent_or_no_children.extend([1 for _ in range(len(children))])
                    edges = [adjmat[parent,idx] for idx in range(adjmat.shape[1]) if adjmat[parent,idx]!=-1]
                    bs_mask.append(True)

                children_idxes_offseted = [idx+sampleidx*ent_padded_nseq for idx in children]
                offset_edges = [idx+sampleidx*spo_padded_nseq for idx in edges]
                idx_belonging = [sampleidx for idx in children] # belong to which sample

                children_idxes_offseted_inbatch.extend(children_idxes_offseted)
                edges_inbatch.extend(edges)
                offset_edges_inbatch.extend(offset_edges)
                children_belongs_to_which_example.extend(idx_belonging)

                nchildrens.append(len(children))

            
            no_parent_or_no_children = torch.as_tensor(no_parent_or_no_children).cuda()
            k_cls_mask = (roi_cls[children_belongs_to_which_example] != -1).float()

            selected_children_atn = torch.index_select(ent_attn.reshape(-1, num_box), 0,
                                                       torch.as_tensor(children_idxes_offseted_inbatch).cuda()) \
                                    * no_parent_or_no_children.reshape((-1,1)) * k_cls_mask # k,nbox
            children_belongs_to_which_example = torch.as_tensor(children_belongs_to_which_example).cuda()

            if self.ctx == 'object_ord':
                offset_edges_inbatch = torch.as_tensor(offset_edges_inbatch).long().cuda()
                k_ctx_table = torch.index_select(ctx_idx_adjusted.view(-1, num_box, num_ctx_box), 0, offset_edges_inbatch) # k,nbox,nctx
                multipled_roi_mask = torch.index_select(roi_mask.view(-1, num_box, num_ctx_box), 0, offset_edges_inbatch) # k,nbox,nctx
            else:
                k_ctx_table = torch.index_select(ctx_idx_adjusted, 0, children_belongs_to_which_example) # k,nbox,nctx
                multipled_roi_mask = torch.index_select(roi_mask, 0, children_belongs_to_which_example) # k,nbox,nctx

            multipled_spo_offset_idx = torch.tensor(np.array(range(k_ctx_table.shape[0])) * num_box, requires_grad=False).cuda().long().reshape((-1,1,1))
            filter_mask = multipled_roi_mask * no_parent_or_no_children.reshape((-1,1,1))

            multipled_spo_offset_idx = (k_ctx_table + multipled_spo_offset_idx).reshape(-1)

            k_scores_children_be_any_ctx = torch.index_select(selected_children_atn.reshape(-1), 0, multipled_spo_offset_idx) # kxnboxxnctx
            k_scores_children_be_any_ctx = k_scores_children_be_any_ctx.reshape((-1, num_box, num_ctx_box)) * filter_mask # some don't have ctx box

            k_scores_rel_by_any_edge = torch.index_select(spo_attn.reshape(-1, num_box, num_ctx_box), 0,
                                                          torch.as_tensor(offset_edges_inbatch).cuda().long()) * filter_mask * k_cls_mask.view(-1,num_box,1 )# k,n,nctx

            if self.bp == "sgmn":
                transfer_obj2sub = torch.sum(k_scores_children_be_any_ctx * k_scores_rel_by_any_edge * multipled_roi_mask
                                             , dim=-1) + 1e-6 # FIXME: why add 1e-20?
                if opt == 'log':
                    transfer_obj2sub = torch.log(transfer_obj2sub)

            else:
                children_prob = torch.sigmoid(k_scores_children_be_any_ctx)
                masked_edge_atn = torch.softmax(k_scores_rel_by_any_edge, multipled_roi_mask, dim=1) * filter_mask
                transfer_obj2sub = (torch.sum(children_prob*masked_edge_atn, dim=-1)+1e-20)

            ###################################
            if self.tracker and opt in ['mul', 'mul_pow', 'log', 'log_exp']: # update multiple children sequentially
                for sampid, batchid in enumerate(children_belongs_to_which_example):
                    if bs_mask[batchid]:
                        refid = parents[batchid]
                        sub_atn = ent_attn[batchid, refid]
                        add_to_sub_ = transfer_obj2sub[sampid]
                        if self.is_sgmn:
                            if opt == 'mul':
                                update_atn = sub_atn * add_to_sub_ * weight_on_children[batchid, refid, :]
                                update_atn_, _  = self.norm_fun(update_atn)
                            elif opt == 'mul_pow':
                                update_atn = sub_atn.clone() * torch.pow(add_to_sub_.clone(), weight_on_children[batchid, refid, :].clone())
                                update_atn_, _  = self.norm_fun(update_atn)
                            elif opt == 'log_exp':
                                update_atn = torch.exp(torch.log(sub_atn.clone()) + torch.log(add_to_sub_.clone()) * weight_on_children[batchid, refid, :].clone())
                                update_atn_, _  = self.norm_fun(update_atn)
                            elif opt == 'log':
                                update_atn = torch.exp(torch.log(sub_atn.clone()) + add_to_sub_.clone()* weight_on_children[batchid, refid, :].clone())
                                update_atn_, _  = self.norm_fun(update_atn)
                            elif opt == 'min':
                                update_atn = torch.minimum(sub_atn.clone(), add_to_sub_.clone() * weight_on_children[batchid, refid, :].clone())
                                update_atn_, _  = self.norm_fun(update_atn)
                            else:
                                update_atn = sub_atn.clone() + add_to_sub_.clone() * weight_on_children[batchid, refid, :].clone()
                                update_atn_, _  = self.norm_fun(update_atn)
                        else:
                            update_atn = sub_atn + add_to_sub_
                            update_atn_, _  = self.norm_fun(update_atn)

                        update_atn_[roi_cls[batchid] == -1] = 1e-6 if opt in ['mul', 'mul_pow', 'log', 'log_exp'] else -1
                        ent_attn[batchid, refid] = update_atn_.clone()
                        ent_tracker[batchid].append(update_atn_.clone())
                        child_tracker[batchid].append(selected_children_atn[sampid])
                        edge_tracker[batchid].append(edges_inbatch[sampid])

                ##########################
            else:
                cnt = 0
                add_to_sub = []
                for sampid, nc in enumerate(nchildrens):
                    if self.bp == "sgmn":
                        if opt == 'min':
                            tmp = torch.min(transfer_obj2sub[cnt:cnt + nc], 0)[0]
                        else:
                            tmp = transfer_obj2sub[cnt:cnt + nc].sum(0) # mean is more reasonable?
                    else:
                        tmp = transfer_obj2sub[cnt:cnt + nc].mean(0)
                    add_to_sub.append(tmp)
                    cnt += nc
                add_to_sub_ = torch.stack(add_to_sub).float() # bs,nbox

                offset_parents_inbatch = [idx if idx!=-1 else 0 for idx in offset_parents_inbatch]
                sub_atn = torch.index_select(ent_attn.reshape((-1, num_box)), 0, torch.as_tensor(offset_parents_inbatch).cuda())

                if self.is_sgmn:
                    if opt == 'mul':
                        update_atn = sub_atn * add_to_sub_ * weight_on_children[range(bs), parents, :]
                        update_atn_, _  = self.norm_fun(update_atn)
                    elif opt == 'mul_pow':
                        update_atn = sub_atn * torch.pow(add_to_sub_, weight_on_children[range(bs), parents, :])
                        update_atn_, _  = self.norm_fun(update_atn)
                    elif opt == 'log_exp':
                        update_atn = torch.exp(torch.log(sub_atn) + torch.log(add_to_sub_) * weight_on_children[range(bs), parents, :])
                        update_atn_, _  = self.norm_fun(update_atn)
                    elif opt == 'log':
                        update_atn = torch.exp(torch.log(sub_atn) + add_to_sub_ * weight_on_children[range(bs), parents, :])
                        update_atn_, _  = self.norm_fun(update_atn)
                    elif opt == 'min':
                        update_atn = torch.minimum(sub_atn, add_to_sub_ * weight_on_children[range(bs), parents, :])
                        update_atn_, _  = self.norm_fun(update_atn)
                    else:
                        update_atn = sub_atn + add_to_sub_ * weight_on_children[range(bs), parents, :]
                        update_atn_, _  = self.norm_fun(update_atn)
                else:
                    update_atn = sub_atn + add_to_sub_
                    update_atn_, _  = self.norm_fun(update_atn)

                update_atn_[roi_cls == -1] = 1e-6 if opt in ['mul', 'mul_pow', 'log', 'log_exp'] else -1
                for sampleid, refid in enumerate(parents):
                    if bs_mask[sampleid]:
                        ent_attn[sampleid, refid] = update_atn_[sampleid].clone()

                if self.save:
                    torch.save(ent_attn, f'./outputs/visualization/202204/{self.save}_in{it}.pth')
                    print(f'{self.save}_in{it} for visualization saved!')

        if self.save: self.save = None

        if self.tracker:
            return ent_attn, (ent_tracker, child_tracker, edge_tracker)
        else:
            return ent_attn


class TopDownPropagate(nn.Module):
    def __init__(self, ctx, bp):
        super(TopDownPropagate, self).__init__()
        self.ctx = ctx
        self.bp = bp
        self.is_sgmn = bp == 'sgmn'
        self.norm_fun = NormAttnMap()

    def forward(self, traversal_lists, adj_matrices, ent_attn, spo_attn, ctx_idx_adjusted, roi_cls, roi_mask, weight_on_children=None, opt='add'):
        bs = ent_attn.shape[0]
        ent_padded_nseq = ent_attn.shape[1]
        spo_padded_nseq = spo_attn.shape[1]
        num_box = ent_attn.shape[2]
        num_ctx_box = spo_attn.shape[-1]
        max_len = max([len(x) for x in traversal_lists])
        padded_traversal_lists = np.stack([np.pad(tl, (max_len-len(tl), 0), constant_values=-1) for tl in traversal_lists])

        for it in range(max_len): # objects are parents, they receive message from subjects
            if self.bp == "none":
                break
            parents = padded_traversal_lists[:,it] # bs, : node index in each sample
            offset_parents_inbatch = [idx+sampleidx*ent_padded_nseq if idx!=-1 else -1 for sampleidx,idx in enumerate(parents)]
            children_idxes_offseted_inbatch = []
            children_belongs_to_which_example = []
            offset_edges_inbatch = []
            nchildrens = []
            no_parent_or_no_children = []
            bs_mask = []
            for sampleidx in range(len(parents)): # for each sample in the batch
                adjmat = adj_matrices[sampleidx]
                parent = int(parents[sampleidx]) # we are sure that all parents must have children
                tmp = [idx for idx in range(adjmat.shape[0]) if adjmat[idx, parent]!=-1]
                if parent==-1 or len(tmp)==0: # when there is only root, it has no children
                    children = [0]
                    no_parent_or_no_children.append(0)
                    edges = [0]
                    bs_mask.append(False)
                else:
                    children = tmp
                    no_parent_or_no_children.extend([1 for _ in range(len(children))])
                    edges = [adjmat[idx, parent] for idx in range(adjmat.shape[0]) if adjmat[idx, parent]!=-1]
                    bs_mask.append(True)

                children_idxes_offseted = [idx+sampleidx*ent_padded_nseq for idx in children]
                offset_edges = [idx+sampleidx*spo_padded_nseq for idx in edges]
                idx_belonging = [sampleidx for _ in children] # belong to which sample
                children_idxes_offseted_inbatch.extend(children_idxes_offseted)
                offset_edges_inbatch.extend(offset_edges)
                children_belongs_to_which_example.extend(idx_belonging)
                nchildrens.append(len(children))

            no_parent_or_no_children = torch.as_tensor(no_parent_or_no_children).cuda()
            k_cls_mask = (roi_cls[children_belongs_to_which_example] != -1).float()

            selected_children_atn = torch.index_select(ent_attn.reshape(-1, num_box), 0,
                                                       torch.as_tensor(children_idxes_offseted_inbatch).cuda()) \
                                    * no_parent_or_no_children.reshape((-1,1)) * k_cls_mask # k,nbox
            children_belongs_to_which_example = torch.as_tensor(children_belongs_to_which_example).cuda()

            if self.ctx == 'object_ord':
                offset_edges_inbatch = torch.as_tensor(offset_edges_inbatch).long().cuda()
                k_ctx_table = torch.index_select(ctx_idx_adjusted.view(-1, num_box, num_ctx_box).transpose(1, 2), 0, offset_edges_inbatch) # k,nbox,nctx
                multipled_roi_mask = torch.index_select(roi_mask.view(-1, num_box, num_ctx_box).transpose(1, 2), 0, offset_edges_inbatch) # k,nbox,nctx
            else:
                k_ctx_table = torch.index_select(ctx_idx_adjusted.transpose(1, 2), 0, children_belongs_to_which_example) # k,nbox,nctx
                multipled_roi_mask = torch.index_select(roi_mask.transpose(1, 2), 0, children_belongs_to_which_example) # k,nbox,nctx

            filter_mask = multipled_roi_mask * no_parent_or_no_children.reshape((-1,1,1))
            k_scores_rel_by_any_edge = torch.index_select(spo_attn.reshape(-1, num_box, num_ctx_box).transpose(1, 2), 0,
                                                          torch.as_tensor(offset_edges_inbatch).cuda().long()) * filter_mask * k_cls_mask.view(-1,1,num_box) # k,n,nctx

            if self.bp == "sgmn":
                transfer_obj2sub = torch.sum(selected_children_atn.unsqueeze(1).repeat(1,5,1) * k_scores_rel_by_any_edge
                                             * multipled_roi_mask, dim=-1) + 1e-6

            else:
                children_prob = torch.sigmoid(selected_children_atn)
                masked_edge_atn = torch.softmax(k_scores_rel_by_any_edge, multipled_roi_mask, dim=1) * filter_mask
                transfer_obj2sub = torch.sum(children_prob*masked_edge_atn, dim=-1)+1e-20
            ##########################
            cnt = 0
            add_to_sub = torch.zeros(bs, num_box).cuda() + 1e-6
            update_mask = k_ctx_table[..., 0]
            for sampid, nc in enumerate(nchildrens):
                tmp = transfer_obj2sub[cnt]
                add_to_sub[sampid, update_mask[sampid]] = tmp
                cnt += nc

            offset_parents_inbatch = [idx if idx!=-1 else 0 for idx in offset_parents_inbatch]
            sub_atn = torch.index_select(ent_attn.reshape((-1, num_box)), 0, torch.as_tensor(offset_parents_inbatch).cuda())

            if self.is_sgmn:
                if opt == 'mul':
                    update_atn = sub_atn * add_to_sub * weight_on_children[range(bs), parents, :]
                    update_atn_, _  = self.norm_fun(update_atn)
                elif opt == 'mul_pow':
                    update_atn = sub_atn * torch.pow(add_to_sub, weight_on_children[range(bs), parents, :])
                    update_atn_, _  = self.norm_fun(update_atn)
                elif opt in ['log_exp', 'log']:
                    update_atn = torch.exp(torch.log(sub_atn) + torch.log(add_to_sub) * weight_on_children[range(bs), parents, :])
                    update_atn_, _  = self.norm_fun(update_atn)
                elif opt == 'min':
                    update_atn = torch.minimum(sub_atn, add_to_sub * weight_on_children[range(bs), parents, :])
                    update_atn_, _  = self.norm_fun(update_atn)
                else:
                    update_atn = sub_atn + add_to_sub * weight_on_children[range(bs), parents, :]
                    update_atn_, _  = self.norm_fun(update_atn)
            else:
                update_atn = sub_atn + add_to_sub
                update_atn_, _  = self.norm_fun(update_atn)

            update_atn_[roi_cls == -1] = 1e-6 if opt in ['mul', 'mul_pow', 'log', 'log_exp'] else -1
            for sampleid, refid in enumerate(parents):
                if bs_mask[sampleid]:
                    ent_attn[sampleid, refid] = update_atn_[sampleid]

        return ent_attn


class AttendRelationModule(nn.Module):
    def __init__(self, dim_vis_feat, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout, vec_sim=False):
        super(AttendRelationModule, self).__init__()
        self.vis_feat_normalizer = NormalizeScale(dim_vis_feat, visual_init_norm)
        self.lfeat_normalizer = NormalizeScale(64, visual_init_norm)
        self.fc = nn.Linear(dim_vis_feat+ 64, jemb_dim)
        self.vis_fc = nn.Linear(dim_vis_feat*2, dim_vis_feat)
        self.spatial_fc = nn.Linear(5, 64)
        self.matching = RelationMatching(jemb_dim, dim_lang_feat, jemb_dim, jemb_dropout, -1, vec_sim=vec_sim)

    def forward(self, subj_feats, obj_feats, rel_feats, lang_feats):
        """

        :param obj_feats: (bs, n, num_cxt, dim_vis_feat)
        :param rel_feats: spatial features (bs, n, num_cxt, 5), 5 is feature dim
        :param lang_feats: (bs, num_seq, dim_lang)
        :return:
        """
        # cxt_feats: (bs, n, num_cxt, dim_vis_feat); cxt_lfeats: (bs, n, num_cxt, 5); lang_feats: (bs, num_seq, dim_lang)
        # compute masks first
        masks = (rel_feats.sum(3) != 0).float()  # bs, n, num_cxt, some box does not have 5 ctx

        # compute joint encoded context
        batch, n, num_cxt = obj_feats.size(0), obj_feats.size(1), obj_feats.size(2)
        vis_feats = self.vis_fc(torch.cat([subj_feats,obj_feats], -1))

        vis_feats = vis_feats.view(-1, vis_feats.shape[-1])
        vis_feats = self.vis_feat_normalizer(vis_feats)
        rel_feats = self.spatial_fc(rel_feats)

        rel_feats = rel_feats.view(-1, rel_feats.shape[-1])
        rel_feats = self.lfeat_normalizer(rel_feats)

        # joint embed
        concat = torch.cat([vis_feats, rel_feats], -1) # bsxnxnctx, dim,
        rel_feats = self.fc(concat)
        num = n
        rel_feats = rel_feats.view(batch, num, num_cxt, -1)  # bs, n, 10, jemb_dim

        attn = self.matching(rel_feats, lang_feats, masks)

        return attn


class RelationMatching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_dropout, min_value=-1, vec_sim=False, dual=False):
        super(RelationMatching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.lang_dim=lang_dim

        self.min_value = min_value
        self.vec_sim = vec_sim
        self.dual = dual

        if self.vec_sim == 'sub' or self.vec_sim == 'mul':
            self.sim_dim = 256
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.sim_tranloc_w = nn.Linear(jemb_dim, self.sim_dim)
            self.sim_eval_w = nn.Linear(self.sim_dim, 1)
            if self.dual:
                self.sim_eval_w_1 = nn.Linear(self.sim_dim, 1)

    def forward(self, vis_input, lang_input, masks):
        # vis_input: (bs, n, num_cxt, vim_dim); lang_input: (bs, num_seq, lang_dim);  mask(bs, n, num_cxt)
        
        pseudo_bs, n, num_cxt = vis_input.size(0), vis_input.size(1), vis_input.size(2)
        bs = lang_input.size(0)
        num_seq = lang_input.size(1)
        vis_emb = self.vis_emb_fc(vis_input.view(pseudo_bs * n * num_cxt, -1))
        lang_emb = self.lang_emb_fc(lang_input.view(bs * num_seq, -1))

        # l2-normalize
        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)
        vis_emb_normalized = vis_emb_normalized.view(pseudo_bs, n, num_cxt, -1)
        if pseudo_bs == bs:
            lang_emb_normalized = lang_emb_normalized.view(bs, num_seq, -1)
        else:
            lang_emb_normalized = lang_emb_normalized.view(pseudo_bs, 1, -1)

        # compute cossim
        if self.vec_sim == 'sub' or self.vec_sim == 'mul':
            lang_i = lang_emb_normalized[:, :, :]
            vis_i = vis_emb_normalized.view(pseudo_bs, n * num_cxt, -1)[:, :, :]
            lang_i_expand = lang_i.repeat(1, n * num_cxt, 1)
            if self.vec_sim == 'mul':
                sim_loc = torch.mul(vis_i, lang_i_expand)
            else:
                sim_loc = torch.pow(torch.sub(vis_i, lang_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
            sim_loc = self.relu(sim_loc)
            sim_i = self.tanh(self.sim_eval_w(sim_loc))
            cossim = sim_i.view(pseudo_bs, 1, -1)
            if self.dual:
                sim_i_1 = self.tanh(self.sim_eval_w_1(sim_loc))
                cossim_1 = sim_i_1.view(pseudo_bs, 1, -1).view(bs, num_seq, n, num_cxt)
        else:
            cossim = torch.bmm(lang_emb_normalized,
                           vis_emb_normalized.view(pseudo_bs, n * num_cxt, -1).transpose(1, 2))  # bs, num_seq, n*num_cxt
        cossim = cossim.view(bs, num_seq, n, num_cxt) # TODO: will this cause error?

        # mask cossim
        if pseudo_bs == bs:
            mask_expand = masks.unsqueeze(1).expand(bs, num_seq, n, num_cxt)
        else:
            mask_expand = masks.reshape(bs, num_seq, n, num_cxt)
        cossim = mask_expand * cossim
        cossim[mask_expand == 0] = self.min_value
        cossim = F.relu(cossim)
        if self.dual:
            cossim_1 = mask_expand * cossim_1
            cossim_1[mask_expand == 0] = self.min_value
            cossim_1 = F.relu(cossim_1)

        if self.dual:
            return (cossim, cossim_1)
        else:
            return cossim


class AttendLocationModule(nn.Module):
    def __init__(self, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout, vec_sim=False, dual_match=False):
        super(AttendLocationModule, self).__init__()
        self.lfeat_normalizer = NormalizeScale(5, visual_init_norm)
        self.fc = nn.Linear(5, jemb_dim)
        self.dual_match = dual_match
        if not dual_match:
            self.matching = Matching(jemb_dim, dim_lang_feat, jemb_dim, jemb_dropout, -1, vec_sim=vec_sim)
        else:
            self.matching = DualMatching(jemb_dim, dim_lang_feat, jemb_dim, jemb_dropout, -1, vec_sim=vec_sim)

    def forward(self, lfeats, lang_feats, cls):
        # lfeats: (bs, n, 5); lang_feats: (bs, num_seq, dim_lang_feat)
        bs, n = lfeats.size(0), lfeats.size(1)

        lfeats = self.lfeat_normalizer(lfeats.view(bs * n, -1))
        loc_feats = self.fc(lfeats).view(bs, n, -1)
        if not self.dual_match:
            attn = self.matching(loc_feats, lang_feats, (cls != -1).float())
            return attn
        else:
            attn_1, attn_2 = self.matching(loc_feats, lang_feats, (cls != -1).float())
            return attn_1, attn_2


class AttendNodeModule(nn.Module):
    def __init__(self, dim_vis_feat, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout, vec_sim=False,
                 dual_match=False):
        super(AttendNodeModule, self).__init__()
        self.dual_match = dual_match
        if not dual_match:
            self.matching = Matching(dim_vis_feat, dim_lang_feat, jemb_dim, jemb_dropout, -1, vec_sim=vec_sim)
        else:
            self.matching = DualMatching(dim_vis_feat, dim_lang_feat, jemb_dim, jemb_dropout, -1, vec_sim=vec_sim)

        self.feat_normalizer = NormalizeScale(dim_vis_feat, visual_init_norm)

    def forward(self, vis_feats, lang_feats, cls):
        bs, n = vis_feats.size(0), vis_feats.size(1)
        num = n
        vis_feats = self.feat_normalizer(vis_feats.view(bs * num, -1)).view(bs, num, -1)

        if not self.dual_match:
            attn = self.matching(vis_feats, lang_feats, (cls != -1).float())
            return attn
        else:
            attn_1, attn_2 = self.matching(vis_feats, lang_feats, (cls != -1).float())
            return attn_1, attn_2


class Matching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_dropout, min_value, vec_sim=False):
        super(Matching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        # fixme: be careful about this, learned padding value in match scores
        self.min_value = min_value

        self.vec_sim = vec_sim
        self.sim_dim = 256
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        if self.vec_sim == 'sub' or self.vec_sim == 'mul':
            self.sim_tranloc_w = nn.Linear(jemb_dim, self.sim_dim)
            self.sim_eval_w = nn.Linear(self.sim_dim, 1)
        elif self.vec_sim == 'mlp':
            self.sim_tranloc_w = nn.Linear(jemb_dim*2, self.sim_dim)
            self.sim_eval_w = nn.Linear(self.sim_dim, 1)
        elif self.vec_sim == 'linear':
            self.sim_tranloc_w = nn.Linear(jemb_dim*2, 1)
        elif self.vec_sim == 'sub_simp':
            self.sim_tranloc_w = nn.Linear(jemb_dim, 1)

    def forward(self, vis_input, lang_input, mask):
        bs, n = vis_input.size(0), vis_input.size(1)
        num_seq = lang_input.size(1)
        vis_emb = self.vis_emb_fc(vis_input.view(bs * n, -1))
        lang_emb = self.lang_emb_fc(lang_input.view(bs * num_seq, -1))

        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)
        vis_emb_normalized = vis_emb_normalized.view(bs, n, -1)
        lang_emb_normalized = lang_emb_normalized.view(bs, num_seq, -1)

        if not self.vec_sim:
            cossim = torch.bmm(lang_emb_normalized, vis_emb_normalized.transpose(1, 2))  # bs, num_seq, n
        elif self.vec_sim == 'sub_simp':
            cossim = []
            for j in range(num_seq):
                lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                vis_i = vis_emb_normalized[:, :, :]
                lang_i_expand = lang_i.repeat(1, n, 1)
                sim_loc = l2norm(torch.pow(torch.sub(vis_i, lang_i_expand), 2), dim=-1)
                sim_loc = self.sim_tranloc_w(sim_loc)
                sim_i = self.tanh(sim_loc)
                cossim.append(sim_i.view(bs, 1, -1))
            cossim = torch.cat(cossim, 1)
        elif self.vec_sim == 'sub' or self.vec_sim == 'mul':
            cossim = []
            for j in range(num_seq):
                lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                vis_i = vis_emb_normalized[:, :, :]
                lang_i_expand = lang_i.repeat(1, n, 1)
                if self.vec_sim == 'mul':
                    sim_loc = torch.mul(vis_i, lang_i_expand)
                else:
                    sim_loc = torch.pow(torch.sub(vis_i, lang_i_expand), 2)
                sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
                sim_loc = self.relu(sim_loc)
                sim_i = self.tanh(self.sim_eval_w(sim_loc))
                cossim.append(sim_i.view(bs, 1, -1))
            cossim = torch.cat(cossim, 1)
        elif self.vec_sim == 'mlp':
            cossim = []
            for j in range(num_seq):
                lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                vis_i = vis_emb_normalized[:, :, :]
                lang_i_expand = lang_i.repeat(1, n, 1)
                sim_loc = torch.cat((vis_i, lang_i_expand), 2)
                sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
                sim_loc = self.relu(sim_loc)
                sim_i = self.tanh(self.sim_eval_w(sim_loc))
                cossim.append(sim_i.view(bs, 1, -1))
            cossim = torch.cat(cossim, 1)
        elif self.vec_sim == 'linear':
            cossim = []
            for j in range(num_seq):
                lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                vis_i = vis_emb_normalized[:, :, :]
                lang_i_expand = lang_i.repeat(1, n, 1)
                sim_loc = l2norm(torch.cat((vis_i, lang_i_expand), 2), dim=-1)
                sim_i = self.tanh(self.sim_tranloc_w(sim_loc))
                cossim.append(sim_i.view(bs, 1, -1))
            cossim = torch.cat(cossim, 1)

        mask_expand = mask.unsqueeze(1).expand(bs, num_seq, n).float()
        cossim_ = cossim.clone()
        cossim_[mask_expand == 0] = self.min_value

        return cossim_


class DualMatching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_dropout, min_value, vec_sim=False):
        super(DualMatching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        # fixme: be careful about this, learned padding value in match scores
        self.min_value = min_value

        self.vec_sim = vec_sim
        if self.vec_sim:
            self.sim_dim = 256
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            if self.vec_sim == 'sub_early':
                self.sim_tranloc_w_1 = nn.Linear(jemb_dim, self.sim_dim)
                self.sim_tranloc_w_2 = nn.Linear(jemb_dim, self.sim_dim)
                self.sim_eval_w = nn.Linear(self.sim_dim, 1)
            elif self.vec_sim == 'sub_repr':
                self.vis_emb_fc_1 = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                                nn.BatchNorm1d(jemb_dim),
                                                nn.ReLU(),
                                                nn.Dropout(jemb_dropout),
                                                nn.Linear(jemb_dim, jemb_dim),
                                                nn.BatchNorm1d(jemb_dim))
                self.lang_emb_fc_1 = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                                 nn.BatchNorm1d(jemb_dim),
                                                 nn.ReLU(),
                                                 nn.Dropout(jemb_dropout),
                                                 nn.Linear(jemb_dim, jemb_dim),
                                                 nn.BatchNorm1d(jemb_dim))
                self.sim_tranloc_w = nn.Linear(jemb_dim, self.sim_dim)
                self.sim_eval_w = nn.Linear(self.sim_dim, 1)
            elif self.vec_sim == 'mlp':
                self.sim_tranloc_w = nn.Linear(jemb_dim*2, self.sim_dim)
                self.sim_eval_w_1 = nn.Linear(self.sim_dim, 1)
                self.sim_eval_w_2 = nn.Linear(self.sim_dim, 1)
            else:
                self.sim_tranloc_w = nn.Linear(jemb_dim, self.sim_dim)
                self.sim_eval_w_1 = nn.Linear(self.sim_dim, 1)
                self.sim_eval_w_2 = nn.Linear(self.sim_dim, 1)

    def forward(self, vis_input, lang_input, mask):
        bs, n = vis_input.size(0), vis_input.size(1)
        num_seq = lang_input.size(1)
        vis_emb = self.vis_emb_fc(vis_input.view(bs * n, -1))
        lang_emb = self.lang_emb_fc(lang_input.view(bs * num_seq, -1))

        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)
        vis_emb_normalized = vis_emb_normalized.view(bs, n, -1)
        lang_emb_normalized = lang_emb_normalized.view(bs, num_seq, -1)

        if not self.vec_sim:
            cossim_1 = torch.bmm(lang_emb_normalized, vis_emb_normalized.transpose(1, 2))  # bs, num_seq, n
            cossim_2 = cossim_1
        else:
            if self.vec_sim == 'sub_repr':
                vis_emb_1 = self.vis_emb_fc_1(vis_input.view(bs * n, -1))
                lang_emb_1 = self.lang_emb_fc_1(lang_input.view(bs * num_seq, -1))
        
                vis_emb_normalized_1 = F.normalize(vis_emb_1, p=2, dim=1)
                lang_emb_normalized_1 = F.normalize(lang_emb_1, p=2, dim=1)
                vis_emb_normalized_1 = vis_emb_normalized_1.view(bs, n, -1)
                lang_emb_normalized_1 = lang_emb_normalized_1.view(bs, num_seq, -1)

                cossim_1 = []
                cossim_2 = []
                for j in range(num_seq):
                    lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                    vis_i = vis_emb_normalized[:, :, :]
                    lang_i_expand = lang_i.repeat(1, n, 1)
                    sim_loc = torch.pow(torch.sub(vis_i, lang_i_expand), 2)
                    sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
                    sim_loc = self.relu(sim_loc)

                    lang_i_1 = lang_emb_normalized_1[:, j, :].view(bs, 1, -1)
                    vis_i_1 = vis_emb_normalized_1[:, :, :]
                    lang_i_expand_1 = lang_i_1.repeat(1, n, 1)
                    sim_loc_1 = torch.pow(torch.sub(vis_i_1, lang_i_expand_1), 2)
                    sim_loc_1 = l2norm(self.sim_tranloc_w(sim_loc_1), dim=-1)
                    sim_loc_1 = self.relu(sim_loc_1)

                    sim_i_1 = self.tanh(self.sim_eval_w(sim_loc))
                    sim_i_2 = self.tanh(self.sim_eval_w(sim_loc_1))
                    cossim_1.append(sim_i_1.view(bs, 1, -1))
                    cossim_2.append(sim_i_2.view(bs, 1, -1))
                cossim_1 = torch.cat(cossim_1, 1)
                cossim_2 = torch.cat(cossim_2, 1)

            else:
                cossim_1 = []
                cossim_2 = []
                for j in range(num_seq):
                    lang_i = lang_emb_normalized[:, j, :].view(bs, 1, -1)
                    vis_i = vis_emb_normalized[:, :, :]
                    lang_i_expand = lang_i.repeat(1, n, 1)
                    if self.vec_sim == 'mul':
                        sim_loc = torch.mul(vis_i, lang_i_expand)
                    elif self.vec_sim == 'mlp':
                        sim_loc = torch.cat((vis_i, lang_i_expand), 2)
                    else: # sub, sub_early
                        sim_loc = torch.pow(torch.sub(vis_i, lang_i_expand), 2)

                    if self.vec_sim == 'sub_early':
                        sim_loc_1 = self.relu(l2norm(self.sim_tranloc_w_1(sim_loc), dim=-1))
                        sim_loc_2 = self.relu(l2norm(self.sim_tranloc_w_2(sim_loc), dim=-1))
                        sim_i_1 = self.tanh(self.sim_eval_w(sim_loc_1))
                        sim_i_2 = self.tanh(self.sim_eval_w(sim_loc_2))
                    else:
                        sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
                        sim_loc = self.relu(sim_loc)
                        sim_i_1 = self.tanh(self.sim_eval_w_1(sim_loc))
                        sim_i_2 = self.tanh(self.sim_eval_w_2(sim_loc))
                    cossim_1.append(sim_i_1.view(bs, 1, -1))
                    cossim_2.append(sim_i_2.view(bs, 1, -1))
                cossim_1 = torch.cat(cossim_1, 1)
                cossim_2 = torch.cat(cossim_2, 1)

        mask_expand = mask.unsqueeze(1).expand(bs, num_seq, n).float()
        cossim_1_ = cossim_1.clone()
        cossim_2_ = cossim_2.clone()
        cossim_1_[mask_expand == 0] = self.min_value
        cossim_2_[mask_expand == 0] = self.min_value

        return cossim_1_, cossim_2_


class NormAttnMap(nn.Module):
    def __init__(self, norm_type='cossim'):
        super(NormAttnMap, self).__init__()
        self.norm_type = norm_type

    def forward(self, attn_map): # normalize over the last dimension
        if self.norm_type != 'cossim':
            norm = torch.max(attn_map, dim=-1, keepdim=True)[0].detach()
        else:
            norm = torch.max(torch.abs(attn_map), dim=-1, keepdim=True)[0].detach()
        norm[norm <= 1] = 1
        attn = attn_map / norm # inf

        return attn, norm


class RegressionHead(nn.Module):
    def __init__(self,lang_dim,vis_dim,joint_dim,loc_dim,visual_init_norm,jemb_dropout):
        super(RegressionHead, self).__init__()
        #todo: should this be more complex
        self.feat_normalizer = NormalizeScale(vis_dim, visual_init_norm)
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, joint_dim),
                                        nn.BatchNorm1d(joint_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(joint_dim, joint_dim),
                                        nn.BatchNorm1d(joint_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, joint_dim),
                                         nn.BatchNorm1d(joint_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(joint_dim, joint_dim),
                                         nn.BatchNorm1d(joint_dim))
        self.box_regre = nn.Linear(joint_dim*2+loc_dim,4)
        self.box2box_transform=Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))

    def forward(self, grd_idx, phrase_feat, vis_feat, locfeat, proposals, gt_coors, phrase_mask, match_mask):
        # todo: lang_feat is avg(spo_atnfeats),change?
        bs, n = vis_feat.size(0), vis_feat.size(1)
        bs_idx=torch.arange(0,bs)

        # merge lang feats
        phrase_num=phrase_mask.sum(dim=1).unsqueeze(-1)
        lang_feats=torch.div(phrase_feat.sum(1),phrase_num)

        # pred box feats
        pred_vis_feat=vis_feat[bs_idx,grd_idx,:]
        vis_feats = self.feat_normalizer(pred_vis_feat)

        # pred box loc feats
        pred_loc_feat=locfeat[bs_idx,grd_idx,:]

        # transform feats
        lang_emb=self.lang_emb_fc(lang_feats)
        vis_emb = self.vis_emb_fc(vis_feats)

        # normalize feats
        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)

        fuse_emb=torch.cat([vis_emb_normalized,pred_loc_feat,lang_emb_normalized,],dim=1)
        pred_anchor_deltas=self.box_regre(fuse_emb)
        pred_coors=torch.tensor(proposals[bs_idx,grd_idx,:]).cuda()

        smooth_l1_beta=0.0

        # only regression for iou>0.5
        batch_ious=self.batch_compute_IoU(gt_coors,pred_coors)
        regre_mask=batch_ious>0.5 # bs,1

        # box coors are (x,y,x1,y1)
        gt_anchor_deltas=self.box2box_transform.get_deltas(src_boxes=pred_coors, target_boxes=gt_coors)
        regression_loss = smooth_l1_loss(
                    pred_anchor_deltas,
                    gt_anchor_deltas,
                    smooth_l1_beta,
                    reduction="none"
                )
        mask = torch.tensor(match_mask).cuda()*regre_mask
        regression_loss=regression_loss.sum(dim=1)*mask
        
        regression_loss_sum=torch.sum(regression_loss)/sum(mask) if sum(mask)!=0 else torch.tensor(0.0).cuda()
        pred_coors_regre=self.box2box_transform.apply_deltas(
                pred_anchor_deltas, pred_coors)


        return regression_loss_sum,pred_coors_regre
    
    def batch_compute_IoU(self, b1, b2):
        # x,y,x1,y1
        iw = torch.min(torch.cat([b1[:,[2]],b2[:,[2]]],dim=1),dim=1)[0] - torch.max(torch.cat([b1[:,[0]],b2[:,[0]]],dim=1),dim=1)[0] + 1
        iw[iw <= 0]=0.0
        ih = torch.min(torch.cat([b1[:,[3]],b2[:,[3]]],dim=1),dim=1)[0] - torch.max(torch.cat([b1[:,[1]],b2[:,[1]]],dim=1),dim=1)[0] + 1
        ih[ih <= 0]=0.0
        ua = (b1[:,2] - b1[:,0] + 1) * (b1[:,3] - b1[:,1] + 1) + (b2[:,2] - b2[:,0] + 1) * (b2[:,3] - b2[:,1] + 1) - iw*ih
        return iw * ih / ua
        

