from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead
from .model import UniterPreTrainedModel, UniterModel


class UniterForNegRE(UniterPreTrainedModel):
    """ Finetune UNITER for NEG_RE
    """
    def __init__(self, config, img_dim, opts, loss="cls",
                 margin=0.2, hard_ratio=0.3, mlp=1, pool="none", pool_with_fuse=False, ng_branch='global',
                 language_only=False, match_only=False, use_prompt=False, itm_match_head=False, oracle='', use_layer=12, fix_layer=0):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, use_layer)
        if use_layer==12 and fix_layer!=0:
            if fix_layer==12:
                print('fix uniter')
                self.uniter=self.uniter.eval()
                for p in self.uniter.parameters(): 
                        p.requires_grad = False
            else:
                print('fix {} uniter layers'.format(fix_layer))
                for p in self.uniter.embeddings.parameters(): 
                    p.requires_grad = False
                for p in self.uniter.img_embeddings.parameters(): 
                    p.requires_grad = False
                
                for i in range(fix_layer):
                    for p in self.uniter.encoder.layer[i].parameters():
                        p.requires_grad=False

                self.uniter.embeddings.eval()
                self.uniter.img_embeddings.eval()
                for i in range(fix_layer):
                    self.uniter.encoder.layer[i].eval()
        

        self.itm_match_head = itm_match_head
        if mlp == 1:
            self.re_output = nn.Linear(config.hidden_size, 1)
        # elif mlp == 2:
        #     self.re_output = nn.Sequential(
        #         nn.Linear(config.hidden_size, config.hidden_size),
        #         GELU(),
        #         LayerNorm(config.hidden_size, eps=1e-12),
        #         nn.Linear(config.hidden_size, 1)
        #     )
        else:
            raise ValueError("MLP restricted to be 1 or 2 layers.")
        self.grd_loss = loss
        assert self.grd_loss in ['cls', 'rank']
        if self.grd_loss == 'rank':
            self.margin = margin
            self.hard_ratio = hard_ratio
        else:
            self.crit = nn.CrossEntropyLoss(reduction='none')
        
        
        if 'fine' in ng_branch:
            self.ng_output_fine = nn.Linear(config.hidden_size, 2)
            self.init_output(self.ng_output_fine,self.itm_match_head)
        if 'global' in ng_branch:
            self.ng_output_global = nn.Linear(config.hidden_size, 2)
            self.init_output(self.ng_output_global,self.itm_match_head)

        self.ng_branch = ng_branch
        self.pooling = pool
        self.pool_fuse=pool_with_fuse
        if self.pool_fuse:
            self.ng_output = nn.Linear(config.hidden_size*2, 2)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(config.hidden_size, 2)
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)
        self.language_only=language_only
        self.match_only=match_only

        self.use_prompt=use_prompt
        self.apply(self.init_weights)
        if self.use_prompt=='fill':
            self.cls = BertOnlyMLMHead(
                config, self.uniter.embeddings.word_embeddings.weight)
        if 'recall' in oracle and 'att' in oracle:
            self.oracle='recall_att'
        elif 'recall' in oracle and 'sim' in oracle:
            self.oracle='recall_sim'
        else:
            self.oracle=oracle


    def init_output(self,layer,itm_match_head):
        """ need to be called after from pretrained """
        pass
        # if itm_match_head:
        #     layer.weight.data = self.itm_output.weight.data
        #     layer.bias.data = self.itm_output.bias.data

    def forward(self, batch, compute_loss=True, detect_foil=False, extract_feat=False, detect_foil_type = 'cos_simi', add_dis=False, extract_layer=[-1], modal='bi', compute_logits='none',device=None):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids'].to(device)
        txt_labels = batch['txt_labels']
        position_ids = batch['position_ids'].to(device)
        img_feat = batch['img_feat'].to(device)
        img_pos_feat = batch['img_pos_feat'].to(device)
        attn_masks = [batch[k] for k in ['attn_masks','attn_masks_txt','attn_masks_img']] #batch['attn_masks'].to(device)
        gather_index = batch['gather_index']
        obj_masks = batch['obj_masks'].to(device)
        txt_masks = batch['txt_masks']

        res={
            'img_hidden':None,
            'txt_hidden':None,
            'match_loss':None,
            'grd_loss':None,
            'match_logits':None,
            'grd_logits':None,
            'all_layer_atts':None
        }

        # only language
        # todo: add language only
        if self.language_only:
            img_feat=torch.zeros_like(img_feat)
            img_pos_feat=torch.zeros_like(img_pos_feat)

        remain_att='recall_att' in self.oracle and not self.training
        if ('recall_sim' in self.oracle and not self.training) or extract_feat:
            output_all_encoded_layers=True
        else:
            output_all_encoded_layers=False
        
        txt_lens, num_bbs = batch["txt_lens"], batch["num_bbs"]

        if extract_feat:
            attn_masks[0]=attn_masks[0].to(device)
            gather_index=gather_index.to(device)
            sequence_output_all,all_layer_atts = self.uniter(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attn_masks[0], gather_index,
                                    output_all_encoded_layers=output_all_encoded_layers,
                                    remain_att=remain_att)

            selected_sequence_output = self.select_layer_feats(sequence_output_all,extract_layer)

            img_hidden = self._get_image_hidden(
            selected_sequence_output, txt_lens, num_bbs, extract_feat=extract_feat)

            txt_hidden=self._get_txt_hidden(
            selected_sequence_output, txt_lens, num_bbs, extract_feat=extract_feat)# bs,maxt_len,hidden

            res.update({
                'img_hidden': img_hidden,
                'txt_hidden': txt_hidden,
            })
        
        if not self.training and 'recall' in self.oracle:
            all_layer_valid_atts=[]
            if 'sim' in self.oracle:
                
                for layer_output in sequence_output_all:
                    layer_sim=self.cos_similar2(layer_output,layer_output)
                    valid_sim=self.txt_to_box_att(layer_sim, txt_lens, num_bbs)
                    all_layer_valid_atts.append(valid_sim)

            # get attention scores in each layer
            else:
                for layer_att in all_layer_atts:
                    valid_att=self.txt_to_box_att(layer_att, txt_lens, num_bbs)
                    all_layer_valid_atts.append(valid_att)

            
            # return img_hidden,txt_hidden# txt_hidden has no special chars in the head and tail
            res.update({
                'all_layer_atts':all_layer_valid_atts
            })
        
        if compute_logits=='none':
            return res

        # select specific layers to compute logits
        layers_for_logits=[9,10,11]
        sequence_output_ori = self.select_layer_feats(sequence_output_all,layers_for_logits)

        # get only the region part
        sequence_output = self._get_image_hidden(
            sequence_output_ori, txt_lens, num_bbs)#, extract_feat=extract_feat)
        
        # ng
        ng_scores_fine=None
        if 'global' in self.ng_branch:
            pooled_output = self.uniter.pooler(sequence_output_ori)
        
        if 'fine' in self.ng_branch:
            
            txt_hidden=self._get_txt_hidden(
            sequence_output_ori, txt_lens, num_bbs)# bs,maxt_len,hidden
            #todo: txt mask
            txt_hidden_valid = self.get_valid_txt(txt_hidden, txt_masks, txt_lens)# cls and sep becomes 0
            
            if self.pool_fuse:
                vis_pool_feat=self.vis_hidden_pool(sequence_output, num_bbs, device).unsqueeze(1)
                vis_pool_feat=torch.cat([vis_pool_feat]*txt_hidden_valid.size(1),dim=1)
                txt_hidden_valid=torch.cat([vis_pool_feat,txt_hidden_valid],dim=2)
            element_prob=self.ng_output_fine(txt_hidden_valid)
            ng_scores_fine=self.element_prob_pool(element_prob, txt_lens, txt_hidden_valid, device)
        # re score (n, max_num_bb)
        scores = self.re_output(sequence_output).squeeze(2)
        # [-1,1]
        scores = torch.tanh(scores)
        scores = scores.masked_fill(obj_masks, -1)  # mask out non-objects -1e4
        res.update({
            'match_logits':ng_scores_fine,
            'grd_logits':scores
        })
        return res
        scores_dict.update({
            'ng_scores_global':ng_scores_global,
            'ng_scores_fine':ng_scores_fine,
            're_scores':scores,
            'foil_scores':None,
            'predict_mask_scores':predict_mask_scores
        })
        
        

        if compute_loss:
            # ng loss
            targets = batch['targets']
            ng_loss_fine=None
            ng_loss_global=None
            if ng_scores_global is not None:
                ng_loss_global=F.cross_entropy(
                    ng_scores_global, targets, reduction='none')

            if ng_scores_fine is not None:
                ng_loss_fine = F.cross_entropy(
                ng_scores_fine, targets, reduction='none')
            
            loss_dict.update({
                'ng_loss_global':ng_loss_global,
                'ng_loss_fine':ng_loss_fine
            })
            
            # re loss
            tgt_box = batch["tgt_box"]
            if self.grd_loss == 'cls':
                ce_loss = self.crit(scores, tgt_box.squeeze(-1))  # (n, ) as no reduction
                loss_dict.update({
                    're_loss':ce_loss
                })
            elif self.grd_loss == 'rank':
                
                # ranking
                _n = len(num_bbs)
                # positive (target)
                pos_ix = tgt_box
                pos_sc = scores.gather(1, pos_ix.view(_n, 1))  # (n, 1)
                pos_sc = torch.sigmoid(pos_sc).view(-1)  # (n, ) sc[0, 1]
                # negative
                neg_ix = self.sample_neg_ix(scores, tgt_box, num_bbs)
                neg_sc = scores.gather(1, neg_ix.view(_n, 1))  # (n, 1)
                neg_sc = torch.sigmoid(neg_sc).view(-1)  # (n, ) sc[0, 1]
                # ranking
                mm_loss = torch.clamp(
                    self.margin + neg_sc - pos_sc, 0)  # (n, )
                loss_dict.update({
                    're_loss':mm_loss
                })
            
            loss_dict['masked_lm_loss']=None
            if self.use_prompt=='fill':
                masked_lm_loss = F.cross_entropy(predict_mask_scores,
                                                targets,
                                                reduction='none')
                loss_dict['masked_lm_loss']=masked_lm_loss
            

            
            if add_dis:
                # weakly supervised loss
                txt_hidden=self._get_txt_hidden(
                sequence_output_ori, txt_lens, num_bbs)
                # pooling
                # todo: box and txt mask
                
                txt_hidden_valid = self.get_valid_txt(txt_hidden, txt_masks, txt_lens)
                
                vis_pool_feat=self.vis_hidden_pool(sequence_output, num_bbs, device)
                txt_pool_feat=self.txt_hidden_pool(txt_hidden_valid, txt_lens, device)
                # vis_pool_feat=sequence_output.mean(1)
                # txt_pool_feat=txt_hidden_valid.mean(1)
                dis_loss=distance_loss(vis_pool_feat,txt_pool_feat)
            else:
                dis_loss=torch.zeros_like(ce_loss)


            loss_dict.update({
                'dis_loss':dis_loss
            })
            if self.match_only:
                
                loss_dict['re_loss']=torch.zeros_like(loss_dict['re_loss'])

            return loss_dict #ng_loss, ce_loss, dis_loss
        
        elif not self.training and detect_foil:
            
            txt_hidden=self._get_txt_hidden(
            sequence_output_ori, txt_lens, num_bbs)# bs,maxt_len,hidden
            if detect_foil_type=='cos_simi':
                
                cos=torch.nn.CosineSimilarity(dim=2, eps=1e-6)
                txt_hidden=F.normalize(txt_hidden,dim=-1, eps=1e-6)
                pooled_output=F.normalize(pooled_output,dim=-1, eps=1e-6)
                neg_scores=cos(txt_hidden, pooled_output.unsqueeze(1))
                neg_scores = neg_scores.masked_fill(txt_masks, 2)
            elif detect_foil_type=='cls' or detect_foil_type=='cls_spo':
                if self.pool_fuse:
                    # vis_pool_feat_broad=torch.cat([vis_pool_feat]*txt_hidden.size(1),dim=1)
                    txt_hidden=torch.cat([vis_pool_feat,txt_hidden],dim=2)
                if 'fine' in self.ng_branch:
                    judge_scores = self.ng_output_fine(txt_hidden)# bs,maxt_len,2
                else:
                    judge_scores = self.ng_output_global(txt_hidden)  # bs,maxt_len,2
                judge_scores = F.softmax(judge_scores,dim=2)# todo: if we should use softmax?
                neg_scores = judge_scores[:,:,0]
                if detect_foil_type=='cls':
                    neg_scores = neg_scores.masked_fill(txt_masks, -1e4)

            scores_dict.update({'foil_scores':neg_scores})

        #todo: complete the choice, compute the simi/attention map
        # vis_hidden=self._get_image_hidden(
        #     sequence_output_ori, txt_lens, num_bbs, extract_feat=True)
        # txt_hidden=self._get_txt_hidden(
        #     sequence_output_ori, txt_lens, num_bbs, extract_feat=True)
        # simi_map=self.compute_simi_map(vis_hidden,txt_hidden)
        # scores_dict.update({'simi_map':simi_map})
        
        return scores_dict #answer_scores, scores, neg_scores

    def select_layer_feats(self,sequence_output_all,extract_layer):
        if len(extract_layer)>1:
            selected_sequence_outputs=[]
            for ext_l in extract_layer:
                selected_sequence_outputs.append(sequence_output_all[ext_l].unsqueeze(0))
            
            selected_sequence_output=torch.cat(selected_sequence_outputs)
            selected_sequence_output=selected_sequence_output.mean(0)
        else:
            selected_sequence_output=sequence_output_all[extract_layer[0]]
        return selected_sequence_output
    
    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
    def get_valid_txt(self, txt_hiddens, txt_masks, txt_lens, use_mask=False):

        temp_mask=torch.zeros_like(txt_hiddens)
        for i in range(len(temp_mask)):
            temp_mask[i][1:txt_lens[i]-1]=1
        
        txt_hiddens=txt_hiddens*temp_mask
        return txt_hiddens

    def txt_hidden_pool(self, txt_hidden, txt_lens, device):
        txt_valid_lens=[i-2 for i in txt_lens]
        txt_lens_tensor = torch.tensor(txt_valid_lens).unsqueeze(-1).to(device)
        pool_txt = torch.div(txt_hidden.sum(dim = 1),txt_lens_tensor)

        return pool_txt

    def vis_hidden_pool(self, vis_hidden, num_bbs, device):
        
        num_bbs_tensor = torch.tensor(num_bbs).unsqueeze(-1).to(device)
        pool_vis = torch.div(vis_hidden.sum(dim = 1),num_bbs_tensor)

        return pool_vis
    

    def element_prob_pool(self, element_prob, txt_lens, txt_hidden_valid, device):
        
        element_prob_sig=element_prob#F.sigmoid(element_prob)
        element_prob_valid=self.get_valid_txt(element_prob_sig, None, txt_lens)
        if self.pooling == 'max':
            #todo: 0 may have negative effect
            
            # global_prob, _ = element_prob.max(dim = 1)
            global_prob_list=[]
            for ep,tl in zip(element_prob_valid,txt_lens):
                per_global_prob,_=ep[1:tl-1].max(dim=0)
                global_prob_list.append(per_global_prob)
            
            global_prob=torch.stack(global_prob_list)
            return global_prob
        elif self.pooling == 'ave':
            txt_valid_lens=[i-2 for i in txt_lens]
            txt_lens_tensor = torch.tensor(txt_valid_lens).unsqueeze(-1).to(device)
            global_prob = torch.div(element_prob_valid.sum(dim = 1),txt_lens_tensor)
            # global_prob = element_prob.mean(dim = 1)
            return global_prob
        elif self.pooling == 'lin':
            global_prob = (element_prob_valid * element_prob_valid).sum(dim = 1) / element_prob_valid.sum(dim = 1)
            return global_prob
        elif self.pooling == 'exp':
            global_prob = (element_prob_valid * element_prob_valid.exp()).sum(dim = 1) / element_prob_valid.exp().sum(dim = 1)
            return global_prob
        elif self.pooling == 'att':
            
            frame_att = F.softmax(self.fc_att(txt_hidden_valid), dim = 1)
            global_prob = (element_prob_valid * frame_att).sum(dim = 1)
            return global_prob

    def sample_neg_ix(self, scores, targets, num_bbs):
        """
        Inputs:
        :scores    (n, max_num_bb)
        :targets   (n, )
        :num_bbs   list of [num_bb]
        return:
        :neg_ix    (n, ) easy/hard negative (!= target)
        """
        neg_ix = []
        cand_ixs = torch.argsort(
            scores, dim=-1, descending=True)  # (n, num_bb)
        for i in range(len(num_bbs)):
            num_bb = num_bbs[i]
            if np.random.uniform(0, 1, 1) < self.hard_ratio:
                # sample hard negative, w/ highest score
                for ix in cand_ixs[i].tolist():
                    if ix != targets[i]:
                        assert ix < num_bb, f'ix={ix}, num_bb={num_bb}'
                        neg_ix.append(ix)
                        break
            else:
                # sample easy negative, i.e., random one
                ix = random.randint(0, num_bb-1)  # [0, num_bb-1]
                while ix == targets[i]:
                    ix = random.randint(0, num_bb-1)
                neg_ix.append(ix)
        neg_ix = torch.tensor(neg_ix).type(targets.type())
        assert neg_ix.numel() == targets.numel()
        return neg_ix

    def compute_simi_map(self,vis_hidden,txt_hidden):
        '''
        input: (bs,box_num,hidden),(bs,txt_len,hidden)
        output:(bs,txt_len,box_num)
        '''
        simi_map=self.cos_similar(txt_hidden,vis_hidden)
        return simi_map

    def cos_similar(self,p: torch.Tensor, q: torch.Tensor, eps=1e-04):
        sim_matrix = p.matmul(q.transpose(-2, -1))
        norm_a = torch.norm(p, p=2, dim=-1)
        norm_b = torch.norm(q, p=2, dim=-1)
        norm_a=self.zero_to_eps(norm_a,eps)
        norm_b=self.zero_to_eps(norm_b,eps)
        sim_matrix /= norm_a.unsqueeze(-1)
        sim_matrix /= norm_b.unsqueeze(-2)
        return sim_matrix
    
    def zero_to_eps(self, tensor2D, eps):
        
        for i in range(len(tensor2D)):
            tensor2D[i][tensor2D[i]==0]=eps
        return tensor2D
    
    def cos_similar2(self,a,b,eps=1e-4):
        a_normalized = F.normalize(a, p=2, dim=-1,eps=eps)
        b_normalized = F.normalize(b, p=2, dim=-1,eps=eps)
        sim_matrix = a_normalized.matmul(b_normalized.transpose(-2, -1))
        return sim_matrix

    def txt_to_box_att(self, all_layer_att, txt_lens, num_bbs):
        '''
        input: per layer att
        output: per layer txt to box att
        '''
        if len(all_layer_att.shape)==4:
            all_layer_att_mean=all_layer_att.mean(1)# bs, all length, all length
        else:
            all_layer_att_mean=all_layer_att
        # txt to all att
        txt_to_all_att = self._get_txt_hidden(all_layer_att_mean, txt_lens, num_bbs, extract_feat=False)# bs, txt len, all length

        # txt to box att
        #todo: need refine
        
        txt_to_all_att_reshape=txt_to_all_att.unsqueeze(-1)
        txt_to_box_att = self._get_box_att(txt_to_all_att_reshape, txt_lens, num_bbs, extract_feat=False)# bs, txt len, box num
        txt_to_box_att=txt_to_box_att.squeeze(-1)
        return txt_to_box_att



    def _get_box_att(self, sequence_output, txt_lens, num_bbs, extract_feat=False):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        max_txt=max(txt_lens)
        max_bb = max(num_bbs)
        
        hid_size = sequence_output.size(-1)
        for seq_out, len_, nbb in zip(sequence_output.split(1, dim=0),
                                      txt_lens, num_bbs):
            img_hid = seq_out[:, :,len_:len_+nbb, :]
            if nbb < max_bb:
                # pad for cos_sim must be -1
                pad = torch.zeros(1, max_txt, max_bb-nbb, hid_size, dtype=img_hid.dtype, device=img_hid.device)-1
                img_hid = torch.cat(
                        [img_hid, pad],
                        dim=2)
            outputs.append(img_hid)

        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden

    def _get_image_hidden(self, sequence_output, txt_lens, num_bbs, extract_feat=False):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        max_bb = max(num_bbs)
        if extract_feat:
            max_bb=126
        hid_size = sequence_output.size(-1)
        for seq_out, len_, nbb in zip(sequence_output.split(1, dim=0),
                                      txt_lens, num_bbs):
            img_hid = seq_out[:, len_:len_+nbb, :]
            if nbb < max_bb:
                img_hid = torch.cat(
                        [img_hid, self._get_pad(
                            img_hid, max_bb-nbb, hid_size)],
                        dim=1)
            outputs.append(img_hid)

        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden
    
    def _get_txt_hidden(self, sequence_output, txt_lens, num_bbs, extract_feat=False):
        """
        Extracting the txt_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - txt_hidden     : (n, max_txt_len, hid_size)
        """
        outputs = []
        max_tl = max(txt_lens)
        # if extract_feat:
        #     max_tl=60 #59
        hid_size = sequence_output.size(-1)
        for seq_out, len_, nbb in zip(sequence_output.split(1, dim=0),
                                      txt_lens, num_bbs):
            if extract_feat:
                txt_hid = seq_out[:, 1:len_-1, :]
                if len_-2 < max_tl:
                    txt_hid = torch.cat(
                            [txt_hid, self._get_pad(
                                txt_hid, max_tl-(len_-2), hid_size)],
                            dim=1)
                
                outputs.append(txt_hid)
            else:
                txt_hid = seq_out[:, :len_, :]
                if len_ < max_tl:
                    txt_hid = torch.cat(
                            [txt_hid, self._get_pad(
                                txt_hid, max_tl-len_, hid_size)],
                            dim=1)
                outputs.append(txt_hid)

        txt_hidden = torch.cat(outputs, dim=0)
        return txt_hidden

    def _get_pad(self, t, len_, hidden_size):
        pad = torch.zeros(1, len_, hidden_size, dtype=t.dtype, device=t.device)
        return pad