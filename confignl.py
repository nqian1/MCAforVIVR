import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')

    #dataset
    parser.add_argument('--dataset', default='rgbn300', help='dataset name: rgbn300]')
    parser.add_argument('--mode', default='v2t', type=str, help='v2t, t2v, mix')
    parser.add_argument('--log_path', default='cross_senlv1', type=str,help='log save path, init:hard mining tri')

    parser.add_argument('--msew', default=2, type=float, help='mse loss weight')
    parser.add_argument('--attloc', default=[0,1,1,0], type=int, help='attention locations')

    parser.add_argument('--gpu', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--imgw', default=256, type=int,metavar='imgw', help='img width')
    parser.add_argument('--imgh', default=256, type=int,metavar='imgh', help='img height')

    parser.add_argument('--modelid', default='epoch_49.t', type=str, help='model for testing')
    parser.add_argument('--batch_size', default=4, type=int, metavar='B', help='training batch size--p')
    parser.add_argument('--num_pos', default=5, type=int,help='num of pos per identity in each modality--k')


    # parser.add_argument('--margin
    # ', default=1, type=float, metavar='margin', help='triplet loss margin')
    parser.add_argument('--epochcount', default=50, type=int, help='epoch count, defult 120')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')


    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--save_epoch', default=10, type=int, metavar='s', help='test every 10 epochs')
    parser.add_argument('--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--test-batch', default=32, type=int,metavar='tb', help='testing batch size')

    parser.add_argument('--seed', default=2023, type=int, metavar='t', help='random seed')
    parser.add_argument('--test-only', action='store_true', help='test only')



    return parser
