# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# from pytorch_transformers import XLNetConfig, XLNetModel  # old-version
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
# from transformers import BertTokenizer, BertModel

class Encoder_XLNet(nn.Module):

    def __init__(self, config, x_embed):
        super().__init__()

        # pretrained_weights = "xlnet-base-cased"
        self.model = XLNetModel.from_pretrained(config.pretrained_weights)
        self.pretrained_config = XLNetConfig.from_pretrained(config.pretrained_weights)

        # if config.use_gpu:
        #   self.model = self.model.to(device=torch.device("cuda"))
        # if config.use_parallel:
        #   self.model = torch.nn.DataParallel(self.model)

        # self.encoder_out_size = 768
        self.encoder_out_size = self.model.config.d_model

        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        encoder_out = []
        self.model.eval()

        with torch.no_grad():
            # print(text_inputs)
            # encoder_out = self.model(text_inputs, None, mask_input)[0]  ## should be tested with input-mask
            # encoder_out = self.model(text_inputs, None)[0]  ## should be tested with input-mask
            # encoder_out = self.model(text_inputs, None, None, attention_mask=mask_input)[0]  
            encoder_out = self.model(text_inputs, attention_mask=mask_input)[0]
            ## input_mask: torch.FloatTensor of shape (batch_size, seq_len)

            encoder_out = encoder_out * mask_input.unsqueeze(2)

        return encoder_out

    #
    def forward_skip(self, x_input, mask, len_seq, mode=""):
        # ''' skip embedding part when embedded input is given '''
        # # mask = mask_input.view(x_input.shape)
        # len_seq_sent_sorted, ind_len_sorted = torch.sort(len_seq,
        #                                                  descending=True)  # ind_len_sorted: (batch_size, num_sents)
        # #
        # sent_x_input_sorted = x_input[ind_len_sorted]
        # # self.model.flatten_parameters()

        # sent_lstm_out, _ = self.model(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)

        # # revert to origin order
        # _, ind_origin = torch.sort(ind_len_sorted)
        # encoder_out = sent_lstm_out[ind_origin]

        # # masking
        # # if self.tokenizer_type.startswith('word'):
        # encoder_out = sent_lstm_out * mask.unsqueeze(2)  # with zero masking
        encoder_out = x_input

        return encoder_out
    # end forward