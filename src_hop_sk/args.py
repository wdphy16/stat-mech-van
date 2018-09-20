import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')
group.add_argument(
    '--model',
    type=str,
    default='sk',
    choices=['hop', 'sk'],
    help='Hamiltonian model')
group.add_argument('--n', type=int, default=10, help='number of sites')
group.add_argument('--beta', type=float, default=1, help='beta = 1 / k_B T')

group = parser.add_argument_group('network parameters')
group.add_argument('--net_depth', type=int, default=1, help='network depth')
group.add_argument('--net_width', type=int, default=1, help='network width')
group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype')
group.add_argument('--bias', action='store_true', help='use bias')
group.add_argument(
    '--z2', action='store_true', help='use Z2 symmetry in sample and loss')
group.add_argument('--res_block', action='store_true', help='use res block')
group.add_argument(
    '--x_hat_clip',
    type=float,
    default=0,
    help='value to clip x_hat around 0 and 1, 0 for disabled')
group.add_argument(
    '--epsilon',
    type=float,
    default=1e-7,
    help='small number to avoid 0 in division and log')

group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer')
group.add_argument(
    '--batch_size', type=int, default=10**3, help='number of samples')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument(
    '--max_step', type=int, default=10**3, help='maximum number of steps')
group.add_argument(
    '--beta_anneal_to', type=float, default=1.5, help='final value of beta')
group.add_argument(
    '--beta_inc', type=float, default=0.2, help='increment of beta')
group.add_argument(
    '--clip_grad',
    type=float,
    default=-1,
    help='gradient normalization, -1 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')
group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument(
    '--print_step',
    type=int,
    default=1,
    help='number of steps to print log, 0 for disabled')
group.add_argument(
    '--save_step',
    type=int,
    default=100,
    help='number of steps to save model, 0 for disabled')
group.add_argument(
    '--visual_step',
    type=int,
    default=100,
    help='number of steps to visualize samples, 0 for disabled')
group.add_argument(
    '--save_sample', action='store_true', help='save samples on print_step')
group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_step, 0 for disabled')
group.add_argument(
    '--print_grad',
    action='store_true',
    help='print summary of gradients for each parameter on visual_step')
group.add_argument(
    '--cuda', type=int, default=0, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')
group.add_argument(
    '--fname',
    type=str,
    default='out.txt',
    help='file name to append the results to')

args = parser.parse_args()
