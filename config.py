import argparse
import math

def get_args():
    parser = argparse.ArgumentParser(description='DeYO exps')

    parser.add_argument('--data_root', default='./data/', help='root for all dataset')
    parser.add_argument('--dset', default='ImageNet-C', type=str, help='ImageNet-C, Waterbirds, ColoredMNIST')
    parser.add_argument('--output', default='./output/dir', help='the output directory of this experiment')
    parser.add_argument('--wandb_interval', default=100, type=int,
                        help='print outputs to wandb at given interval.')
    parser.add_argument('--wandb_log', default=0, type=int)

    parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--continual', default=False, type=bool, help='continual tta or fully tta')

    # dataloader
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=64, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--eata_fishers', default=1, type=int)
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.') # 2000 500
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss') # 2000 100 5000 1
    parser.add_argument('--e_margin', type=float, default=0.4, help='entropy margin E_0 for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='deyo', type=str, help='no_adapt, tent, eata, sar, deyo')
    parser.add_argument('--model', default='resnet50_bn_torch', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm or resnet18_bn')
    parser.add_argument('--exp_type', default='normal', type=str, help='normal, mix_shifts, bs1, label_shifts, spurious')
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=0.4, type=float, help='the threshold for reliable minimization in SAR.')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).')

    # DeYO parameters
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)

    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')
    
    parser.add_argument('--fishers', default=0, type=int)
    parser.add_argument('--filter_ent', default=1, type=int)
    parser.add_argument('--filter_plpd', default=1, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=1, type=int)

    parser.add_argument('--topk', default=1000, type=int)
    
    parser.add_argument('--wbmodel_name', default='waterbirds_pretrained_model.pickle', type=str, help='Waterbirds pre-trained model path')
    parser.add_argument('--cmmodel_name', default='ColoredMNIST_model.pickle', type=str, help='ColoredMNIST pre-trained model path')
    parser.add_argument('--lr_mul', default=1, type=float, help='5 for Waterbirds, ColoredMNIST')

    return parser.parse_args()
