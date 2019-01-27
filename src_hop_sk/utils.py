import os
from glob import glob

import numpy as np
import torch

from args import args

if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

args.out_filename = None


def get_ham_args_features():
    ham_args = '{ham}_n{n}_beta{beta:g}'
    ham_args = ham_args.format(**vars(args))

    features = 'nd{net_depth}_nw{net_width}_made'

    if args.bias:
        features += '_bias'
    if args.z2:
        features += '_z2'
    if args.res_block:
        features += '_res'
    if args.x_hat_clip:
        features += '_xhc{x_hat_clip:g}'

    if args.optimizer != 'adam':
        features += '_{optimizer}'
    if args.beta_anneal_to:
        features += '_ba{beta_anneal_to:g}'
    if args.clip_grad:
        features += '_cg{clip_grad:g}'

    features = features.format(**vars(args))

    return ham_args, features


def init_out_filename():
    if not args.out_dir:
        return
    ham_args, features = get_ham_args_features()
    template = '{args.out_dir}/{ham_args}/{features}/out{args.out_infix}'
    args.out_filename = template.format(**{**globals(), **locals()})


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_dir:
        return
    init_out_filename()
    ensure_dir(args.out_filename)
    if args.save_step:
        ensure_dir(args.out_filename + '_save/')
    if args.visual_step:
        ensure_dir(args.out_filename + '_img/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def my_err(s):
    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn('{} = {}'.format(k, v))
    print_fn('')


def parse_checkpoint_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    if not (args.out_filename and args.save_step):
        return -1
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    if not filename_list:
        return -1
    step = max([parse_checkpoint_name(x) for x in filename_list])
    return step


def clear_checkpoint():
    if not (args.out_filename and args.save_step):
        return
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    for filename in filename_list:
        os.remove(filename)
