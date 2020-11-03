# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import numpy as np
# import torch.distributions.normal as normal
import logging

import w2vEmbReader

from models.encoders.encoder_main import Encoder_Main


import models.model_base
import utils
from utils import FLOAT, LONG

import torch.nn.utils.weight_norm as weightNorm

import fairseq.modules as fairseq

# from apex.normalization.fused_layer_norm import FusedLayerNorm

logger = logging.getLogger()

class Coh_Model_SentAvg(models.model_base.BaseModel):
    def __init__(self, config, corpus_target, embReader):
        """ class for simple baseline submitted to COLING20
        Title: Context-aware Lexical Coherence Modeling
        Ref: 
        """
        super().__init__(config)

        ####
        # init parameters
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.batch_size = config.batch_size

        self.avg_len_doc = config.avg_len_doc

        self.corpus_target = config.corpus_target
        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.pad_id = corpus_target.pad_id
        self.num_special_vocab = corpus_target.num_special_vocab

        self.dropout_rate = config.dropout
        self.rnn_cell_size = config.rnn_cell_size
        self.num_layers = 1
        self.output_size = config.output_size  # the number of final output class
        self.pad_level = config.pad_level

        self.use_gpu = config.use_gpu
        self.gen_logs = config.gen_logs

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        ########
        #
        self.encoder_base = Encoder_Main(config, embReader)

        #
        self.sim_cosine = torch.nn.CosineSimilarity(dim=2)

        #
        fc_in_size = self.encoder_base.encoder_out_size

        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        self.linear_1 = nn.Linear(fc_in_size, linear_1_out)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_2 = nn.Linear(linear_1_out, linear_2_out)
        nn.init.xavier_uniform_(self.linear_2.weight)

        self.linear_out = nn.Linear(linear_2_out, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)
        # nn.init.xavier_normal_(self.linear_out.weight)

        #
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.softmax = nn.Softmax(dim=1)


        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, len_para=None, mode=""):
        batch_size = text_inputs.size(0)

        # #
        if self.pad_level == "sent" or self.pad_level == "sentence":
            text_inputs = text_inputs.view(batch_size, text_inputs.size(1)*text_inputs.size(2))

        #### word level encoding
        encoder_out = self.encoder_base(text_inputs, mask_input, len_seq)

        #### sentence representations
        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        sent_repr = torch.zeros(batch_size, self.max_num_sents, self.encoder_base.encoder_out_size)
        sent_repr = utils.cast_type(sent_repr, FLOAT, self.use_gpu)
        for cur_ind_doc in range(batch_size):
            list_sent_len = len_sents[cur_ind_doc]
            cur_sent_num = int(num_sents[cur_ind_doc])
            cur_loc_sent = 0
            list_cur_doc_sents = []

            for cur_ind_sent in range(cur_sent_num):
                cur_sent_len = int(list_sent_len[cur_ind_sent])
                cur_sent_repr = torch.div(torch.sum(encoder_out[cur_ind_doc, cur_loc_sent:cur_loc_sent+cur_sent_len], dim=0), cur_sent_len)  # avg version

                cur_sent_repr = cur_sent_repr.view(1, 1, -1)  # restore to (1, 1, xrnn_cell_size)
                
                list_cur_doc_sents.append(cur_sent_repr)
                cur_loc_sent = cur_loc_sent + cur_sent_len

            # end for cur_len_sent

            cur_sents_repr = torch.stack(list_cur_doc_sents, dim=1)  # (batch_size, num_sents, rnn_cell_size)
            cur_sents_repr = cur_sents_repr.squeeze(2)

            sent_repr[cur_ind_doc, :cur_sent_num, :] = cur_sents_repr
        # end for cur_doc
        
        # encoder sentence 
        mask_sent = torch.arange(self.max_num_sents, device=num_sents.device).expand(len(num_sents), self.max_num_sents) < num_sents.unsqueeze(1)
        mask_sent = utils.cast_type(mask_sent, FLOAT, self.use_gpu)
        num_sents = utils.cast_type(num_sents, FLOAT, self.use_gpu)
        encoded_sent = sent_repr

        # get averaging
        ilc_vec_sent = torch.div(torch.sum(encoded_sent, dim=1), num_sents.unsqueeze(1))  # (batch_size, rnn_cell_size)

        fc_out = self.linear_1(ilc_vec_sent)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_2(fc_out)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_out(fc_out)
        
        if self.output_size == 1:
            fc_out = self.sigmoid(fc_out)

        # if self.gen_logs:
        #     return fc_out, stacked_std  # for the error analysis
        # else:
        return fc_out


    # end forward
