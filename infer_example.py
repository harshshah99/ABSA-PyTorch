# -*- coding: utf-8 -*-
# file: infer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn.functional as F
import argparse

from data_utils import build_tokenizer, build_embedding_matrix
from atae_lstm import ATAE_LSTM


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=self.tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts,aspect_tokens):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in raw_texts]
        aspect_seqs = [self.tokenizer.text_to_sequence(asp.lower().strip()) for asp in aspect_tokens]
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)
        print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE  ",aspect_indices)

        t_inputs = [context_indices, aspect_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    model_classes = {'atae_lstm': ATAE_LSTM}
    # set your trained models here
    model_state_dict_paths = {
        'atae_lstm': 'state_dict/atae_lstm_restaurant_val_acc0.7554',
        'ian': 'state_dict/ian_restaurant_acc0.7911',
        'memnet': 'state_dict/memnet_restaurant_acc0.7911',
        'aoa': 'state_dict/aoa_restaurant_acc0.8063',
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'atae_lstm'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'restaurant'
    opt.dataset_file = {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    }
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 80
    opt.polarities_dim = 3 #change as required
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    t_probs = inf.evaluate(raw_texts = ['The taste was amazing ', 'the fajita we tried was tasteless and burnt and the sauce was way too sweet.', ' This service is terrible but the taste was good' , 'they have one of the fastest delivery times in the city'] , aspect_tokens=['taste' , 'food', 'taste' , 'service'])
    if(opt.polarities_dim==3):
        print(t_probs.argmax(axis=-1) - 1)
    else:
        print(t_probs.argmax(axis=-1))