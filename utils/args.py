#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    arg_parser.add_argument('--name', type=str, required=True, help='Name of the experiment')

    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--ckpt', default=None, help='checkpoint path')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    arg_parser.add_argument('--decay_step', nargs="+", type=int, default=[50], help='learning rate which milestones to decay')
    arg_parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', help='root of data', default='bert',
                                choices=['bert', 'macbert', 'robert', 'LSTM', 'GRU', 'RNN'])
    arg_parser.add_argument('--decoder_cell', default='FNN', choices=['FNN', 'LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')

    #### BERT Model ####
    arg_parser.add_argument('--lock_bert', action='store_true', help='Lock BERT parameters')
    arg_parser.add_argument('--lock_bert_ratio', type=float, default=0, help='Lock ratio of BERT parameters')
    arg_parser.add_argument('--use_bert_state', type=str, choices=['mean', 'last', 'fuse'], default='last', help='Use mean/last/fuse hidden-state')
    return arg_parser