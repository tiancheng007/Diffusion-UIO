"""This file contains the settings for the training and inference"""
import argparse

def general_settings():
    ### Dataset settings
        # Sizes
    parser = argparse.ArgumentParser(prog = 'UIONet',\
                                     description = 'Dataset, training and network parameters')
    parser.add_argument('--T', type=int, default=10, metavar='length',
                        help='input sequence length')
    parser.add_argument('--T_test', type=int, default=10, metavar='test-length',
                        help='input test sequence length')


    ### Training settings
    parser.add_argument('--use_cuda', type=bool, default=True, metavar='CUDA',
                        help='if True, use CUDA')
    parser.add_argument('--n_steps', type=int, default=100, metavar='N_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_batch', type=int, default=8, metavar='N_batch',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.3, metavar='alpha',
                        help='input alpha [0,1] for the composition loss')

    parser.add_argument("--dataset", type=str, default='./data/xx.npz', help="Dataset file")
    parser.add_argument("--experiment_name", type=str, default='240924_UIONet_run1', help="Name for experiment")


    args = parser.parse_args()
    return args
