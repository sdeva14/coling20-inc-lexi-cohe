# -*- coding: utf-8 -*-

#
import os
import argparse
import logging
import time

#
import numpy as np
import torch
import torch.nn as nn

# written codes
import build_config
import utils

import corpus.corpus_gcdc
import corpus.corpus_toefl

import w2vEmbReader

from models.optim_hugging import AdamW, WarmupLinearSchedule, WarmupCosineSchedule

import torch.nn.functional as F
import torch

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import training

from evaluators import eval_acc, eval_qwk, eval_fscore

from models.model_CoNLL17_Essay import Model_CoNLL17_Essay
from models.model_EMNLP18_Centt import Model_EMNLP18_Centt

from models.model_inc_lexi import Coh_Model_Inc_Lexi
from models.model_SentAvg import Coh_Model_SentAvg
from models.model_DocAvg import Coh_Model_DocAvg

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from corpus.dataset_gcdc import Dataset_GCDC
from corpus.dataset_toefl import Dataset_TOEFL

########################################################

# global parser for arguments
parser = argparse.ArgumentParser()
arg_lists = []

###########################################
###########################################

#
def get_w2v_emb(config, corpus_target):
    embReader = w2vEmbReader.W2VEmbReader(config=config, corpus_target=corpus_target)
    
    return embReader                                              

#
def get_corpus_target(config):
    corpus_target = None
    logger = logging.getLogger()

    if config.corpus_target.lower() == "gcdc":
        logger.info("Corpus: GCDC")
        corpus_target = corpus.corpus_gcdc.CorpusGCDC(config)
    elif config.corpus_target.lower() == "toefl":
        logger.info("Corpus: TOEFL")
        corpus_target = corpus.corpus_toefl.CorpusTOEFL(config)

    return corpus_target
# end get_corpus_target

#
def get_dataset(config, id_corpus, pad_id):
    dataloader_train = None
    dataloader_valid = None
    dataloader_test = None

    if config.corpus_target.lower() == "gcdc":
        dataset_train = Dataset_GCDC(id_corpus["train"], config, pad_id)
        dataset_valid = None
        dataset_test = Dataset_GCDC(id_corpus["test"], config, pad_id)
    elif config.corpus_target.lower() == "toefl":
        dataset_train = Dataset_TOEFL(id_corpus["train"], config, pad_id)
        dataset_valid = Dataset_TOEFL(id_corpus["valid"], config, pad_id)
        dataset_test = Dataset_TOEFL(id_corpus["test"], config, pad_id)

    return dataset_train, dataset_valid, dataset_test

#
def get_model_target(config, corpus_target, embReader):
    model = None
    logger = logging.getLogger()

    if config.target_model.lower().startswith("emnlp18"):
        logger.info("Model: EMNLP18")
        model = Model_EMNLP18_Centt(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower().startswith("conll17"):
        logger.info("Model: CoNLL17")
        model = Model_CoNLL17_Essay(config=config, corpus_target=corpus_target, embReader=embReader)

    elif config.target_model.lower() == "inc_lexi":
        logger.info("Model: Incremental Lexical Coherence")
        model = Coh_Model_Inc_Lexi(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower() == "sent_avg":
        logger.info("Model: Sent_Avg")
        model = Coh_Model_SentAvg(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower() == "doc_avg":
        logger.info("Model: Doc_Avg")
        model = Coh_Model_DocAvg(config=config, corpus_target=corpus_target, embReader=embReader)

    return model

#
def get_optimizer(config, model, len_trainset):
    # basic style
    model_opt = model.module if hasattr(model, 'module') else model  # take care of parallel
    optimizer = model_opt.get_optimizer(config)

    optimizer = model.get_optimizer(config)
    scheduler = None
    
    return optimizer, scheduler


# #
    
#
def exp_model(config):
    ## Pre-processing

    # read corpus then generate id-sequence vector
    corpus_target = get_corpus_target(config)  # get corpus class
    corpus_target.read_kfold(config)

    # get embedding class
    embReader = get_w2v_emb(config, corpus_target)

    # update config depending on environment
    config.max_num_sents = corpus_target.max_num_sents  # the maximum number of sentences in document (i.e., document length)
    config.max_len_sent = corpus_target.max_len_sent  # the maximum length of sentence (the number of words)
    # config.max_len_doc = corpus_target.max_len_doc  # the maximum length of document (the number of words)

    # convert to id-sequence for given k-fold
    cur_fold = config.cur_fold
    # id_corpus, max_len_doc, avg_len_doc = corpus_target.get_id_corpus(cur_fold)
    id_corpus, max_len_doc, avg_len_doc, max_num_para, max_num_sents = corpus_target.get_id_corpus(cur_fold)
    config.max_len_doc = max_len_doc
    config.avg_len_doc = avg_len_doc

    config.avg_num_sents = corpus_target.avg_num_sents
    config.max_num_para = max_num_para

    if config.use_paragraph:
        config.max_num_sents = max_num_sents

    ## Model
    # prepare batch form
    # batch_data_train, batch_data_valid, batch_data_test = get_batch_loader(config, id_corpus)  # get batch-loader class
    dataset_train, dataset_valid, dataset_test = get_dataset(config, id_corpus, embReader.pad_id)

    #### prepare model
    if torch.cuda.is_available():   config.use_gpu = True
    else: config.use_gpu = False
    model = get_model_target(config, corpus_target, embReader)  # get model class
    optimizer, scheduler = get_optimizer(config, model, len(dataset_train))

    # if config.use_parallel:
    #       model = nn.DataParallel(model)
    #       #if not config.use_apex:
    #       optimizer = model.module.get_optimizer(config)
    # if config.use_gpu:
    #       model.to(device=torch.device("cuda"))

    # # if config.use_apex:
    # #     model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    device = "cuda"
    if config.local_rank == -1 or not config.use_gpu:  # when it is not distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.n_gpu = torch.cuda.device_count()  # will be 1 or 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # if config.use_parallel:
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        config.world_size = torch.distributed.get_world_size()
        # config.n_gpu = 1

    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
        #                                                  output_device=config.local_rank,
        #                                                  find_unused_parameters=True)
        # model = apex.parallel.DistributedDataParallel(model)
        # config.n_gpu = 1
    elif config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        optimizer = model.module.get_optimizer(config)

    if config.use_gpu:
        model.to(device)
    
    #### run training and evaluation
    ##
    evaluator = None
    if config.eval_type.lower() == "fscore":
            evaluator = eval_fscore.Eval_Fsc(config)
    elif config.eval_type.lower() == "accuracy":
            evaluator = eval_acc.Eval_Acc(config)
    elif config.eval_type.lower() == "qwk":
            min_rating, max_rating = corpus_target.score_ranges[corpus_target.prompt_id_test]  # in case of MSELoss
            #min_rating, max_rating = corpus_target.score_ranges[corpus_target.cur_prompt_id]  # in case of asap corpus
            evaluator = eval_qwk.Eval_Qwk(config, min_rating, max_rating, corpus_target) 
    ##
    final_eval_best = training.train(model,
                        optimizer,
                        scheduler,
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        dataset_test=dataset_test,
                        config=config,
                        evaluator=evaluator)

    return final_eval_best
# end exp_model

###################################################

if __name__=='__main__':
    ## prepare config
    build_config.process_config()
    config, _ = build_config.get_config()
    utils.prepare_dirs_loggers(config, os.path.basename(__file__))
    logger = logging.getLogger() 
    
    # torch.manual_seed(args.seed)  # set the random seed manually for reproducbility

    ## option configure
    # change pad level to sent if EMNLP18, because of sentence level handling
    if config.target_model.lower() == "emnlp18" \
        or config.target_model.lower() == "ilcr_doc_stru":
        config.pad_level = "sent"

    # automatically extract target corpus from dataset path
    if len(config.corpus_target) == 0:
        cur_corpus_name = os.path.basename(os.path.normpath(config.data_dir))
        config.corpus_target = cur_corpus_name

    # domain information for printing
    cur_domain_train = None
    cur_domain_test = None
    if config.corpus_target.lower() == "toefl":
        cur_domain_train = config.essay_prompt_id_train
        cur_domain_test = config.essay_prompt_id_test
    elif config.corpus_target.lower() == "gcdc":
        cur_domain_train = config.gcdc_domain
        cur_domain_test = config.gcdc_domain

    ## Run model
    list_cv_attempts=[]
    target_attempts = config.cv_attempts
    
    if config.cur_fold > -1:  # test for specific fold
        if cur_domain_train is not None:
            logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
        eval_best_fold = exp_model(config)
        logger.info("{}-fold eval {}".format(config.cur_fold, eval_best_fold))
    else:
        for cur_attempt in range(target_attempts):  # CV only works when whole k-fold eval mode

            ##
            logger.info("Whole k-fold eval mode")
            list_eval_fold = []
            for cur_fold in range(config.num_fold):
                config.cur_fold = cur_fold
                if cur_domain_train is not None:
                    logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
                cur_eval_best_fold = exp_model(config)
                list_eval_fold.append(cur_eval_best_fold)
            
            avg_cv_eval = sum(list_eval_fold) / float(len(list_eval_fold))
            logger.info("Final k-fold eval {}".format(avg_cv_eval))
            logger.info(list_eval_fold)

            list_cv_attempts.append(avg_cv_eval)

    #
    if target_attempts > 1 and len(list_cv_attempts) > 0:
        avg_cv_attempt = sum(list_cv_attempts) / float(len(list_cv_attempts))
        logger.info("Final CV exp result {}".format(avg_cv_attempt))
        logger.info(list_cv_attempts)

        for cur_score in list_cv_attempts:
            print(cur_score)


