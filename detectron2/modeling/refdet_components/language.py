import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import detectron2.utils.refdet_basics as basics
import pickle as pkl

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm',
                 variable_lengths=True, pretrain=False, explicit_dropout=False):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        if pretrain is True:
            embedding_mat = np.load('./data/word_embedding/embed_matrix.npy')
            self.embedding = nn.Embedding.from_pretrained(basics.to_torch(embedding_mat).cuda(), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.output_dropout = nn.Dropout(dropout_p)
        self.explicit_dropout = explicit_dropout # due to pytorch version mismatch, dropout for 1layer lstm is disabled

    def forward(self, input_labels, pretrain_feats):
        
        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1) # bs,nseq, the length is the actual length of each seq before padding

            # make ixs
            input_lengths_list = basics.to_numpy(input_lengths).tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() #rank max to min
            max_length = sorted_input_lengths_list[0]
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # rank-idx
            s2r = {s: r for r, s in enumerate(sort_ixs)} # idx-rank
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]# recover the original order

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long().cuda()
            recover_ixs = input_labels.data.new(recover_ixs).long().cuda()

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs, 0:max_length].long().cuda() # ordered by rank
            assert max(input_lengths_list) == input_labels.size(1)
        
        embedded = self.embedding(input_labels)# all words num,embed size
        embedded = self.input_dropout(embedded)
        
        
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:
            # embedded (batch, seq_len, word_embedding_size)
            ### why pad a second time
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)# pad and pack
            embedded = embedded[recover_ixs] # totalseq, nword, 300

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs] # totalseq, nword, 1024
            if self.explicit_dropout: output = self.output_dropout(output)
            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]# h,c
            
            # merge two results of bi-lstm
            hidden = hidden[:, recover_ixs, :] # what is first dim?
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1) # totalseq, nword

        return output, hidden, embedded, max_length


class RNNEncoderForPretrain(nn.Module):
    def __init__(self, word_embedding_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm',
                 variable_lengths=True, explicit_dropout=False, lang_encode_type='default'):
        super(RNNEncoderForPretrain, self).__init__()
        self.variable_lengths = variable_lengths
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.output_dropout = nn.Dropout(dropout_p)
        self.explicit_dropout = explicit_dropout # due to pytorch version mismatch, dropout for 1layer lstm is disabled
        self.lang_encode_type=lang_encode_type
        if self.lang_encode_type=='simple':
            self.hidden_fc=nn.Linear(word_embedding_size,hidden_size*(2 if bidirectional else 1))

    def forward(self, input_labels, pretrain_feats):

        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1) # bs,nseq, the length is the actual length of each seq before padding

            # make ixs
            input_lengths_list = basics.to_numpy(input_lengths).tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() #rank max to min
            max_length = sorted_input_lengths_list[0]
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # rank-idx
            s2r = {s: r for r, s in enumerate(sort_ixs)} # idx-rank
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long().cuda()
            recover_ixs = input_labels.data.new(recover_ixs).long().cuda()
            
            
            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs, 0:max_length].long().cuda() # ordered by rank
            pretrain_feats = pretrain_feats[sort_ixs, 0:max_length].cuda()
            assert max(input_lengths_list) == input_labels.size(1)

        if self.lang_encode_type=='simple':
            # todo:simpler output
            embedded = pretrain_feats
            # embedded = self.input_dropout(embedded)
            sum_emb = embedded.sum(1)
            mean_emb = torch.div(sum_emb,torch.tensor(sorted_input_lengths_list).unsqueeze(-1).cuda())
            hidden = self.hidden_fc(mean_emb)
            hidden = hidden[recover_ixs]
            embedded = embedded[recover_ixs]
            output = None
        else:
            embedded = pretrain_feats
            embedded = self.input_dropout(embedded)
            
            # todo:substitute this embedding
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

            # forward rnn
            output, hidden = self.rnn(embedded)

            # recover
            if self.variable_lengths:
                ### why pad a second time
                embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
                embedded = embedded[recover_ixs] # totalseq, nword, 300

                # recover rnn
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
                output = output[recover_ixs] # totalseq, nword, 1024
                if self.explicit_dropout: output = self.output_dropout(output)
                # recover hidden
                if self.rnn_type == 'lstm':
                    hidden = hidden[0]
                hidden = hidden[:, recover_ixs, :] # what is first dim?
                hidden = hidden.transpose(0, 1).contiguous()
                hidden = hidden.view(hidden.size(0), -1) # totalseq, nword

        return output, hidden, embedded, max_length


class ModuleInputAttention(nn.Module):
    def __init__(self, input_dim):
        super(ModuleInputAttention, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, word_ctxfeats, word_embfeats, input_labels):
        # context(bs, num_seq, max_length, dim_cxt); input_labels(bs, num_seq, max_length)
        bs, num_seq, max_length = word_ctxfeats.size(0), word_ctxfeats.size(1), word_ctxfeats.size(2)
        word_ctxfeats = word_ctxfeats.view(bs * num_seq, max_length, -1)
        word_embfeats = word_embfeats.view(bs * num_seq, max_length, -1)
        input_labels = input_labels.view(bs*num_seq, max_length)

        aspect_scores = self.fc(word_ctxfeats).squeeze(2)
        attn = F.softmax(aspect_scores, dim=1) # bs, num_seq, max_length

        # mask zeros
        is_not_zero = (input_labels != 0).float()
        attn = attn * is_not_zero
        # want to remove zeros?
        # todo: check this, is the following necessary?
        attn_sum = attn.sum(1).unsqueeze(1).expand(attn.size(0), attn.size(1))
        attn[attn_sum!=0] = attn[attn_sum!=0] / attn_sum[attn_sum!=0]

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)
        
        weighted_emb = torch.bmm(attn3, word_embfeats)
        
        weighted_emb = weighted_emb.squeeze(1)

        weighted_emb = weighted_emb.view(bs, num_seq, -1)
        

        return weighted_emb, attn

