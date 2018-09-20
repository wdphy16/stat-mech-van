#!/usr/bin/env python3
#
# Sample from a trained model

import time

import torch
from numpy import abs, sqrt
from uncertainties import ufloat

import ising
from args import args
from made import MADE
from pixelcnn import PixelCNN
from utils import get_model_args_features, ignore_param

# Set args here or through CLI to match the state
args.model = 'afm'
args.lattice = 'tri'
args.boundary = 'periodic'
args.L = 4
args.beta = 2

args.net = 'pixelcnn'
args.net_depth = 3
args.net_width = 64
args.half_kernel_size = 3
args.bias = True
args.beta_anneal = 0.998

args.batch_size = 10**2
args.max_step = 10**2
args.print_step = 1
state_dir = 'out'

model_args, features = get_model_args_features()
state_filename = '{state_dir}/{model_args}/{features}/out{args.out_infix}_save/10000.state'.format(
    **locals())


def get_mean_err(count, x_sum, x_sqr_sum):
    x_mean = x_sum / count
    x_sqr_mean = x_sqr_sum / count
    x_std = sqrt(abs(x_sqr_mean - x_mean**2))
    x_err = x_std / sqrt(count)
    x_ufloat = ufloat(x_mean, x_err)
    return x_ufloat


if __name__ == '__main__':
    if args.net == 'made':
        net = MADE(**vars(args))
    elif args.net == 'pixelcnn':
        net = PixelCNN(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
    net.to(args.device)
    print('{}\n'.format(net))

    print(state_filename)
    state = torch.load(state_filename, map_location=args.device)
    ignore_param(state['net'], net)
    net.load_state_dict(state['net'])

    F_sum = 0
    F_sqr_sum = 0
    S_sum = 0
    S_sqr_sum = 0
    E_sum = 0
    E_sqr_sum = 0
    start_time = time.time()
    for step in range(args.max_step):
        with torch.no_grad():
            sample, x_hat = net.sample(args.batch_size)
            log_prob = net._log_prob(sample, x_hat)
            energy = ising.energy(sample, args.model, args.lattice,
                                  args.boundary) / args.L**2
            free_energy = energy + 1 / args.beta * log_prob / args.L**2
            entropy = -log_prob / args.L**2

            F_sum += free_energy.sum().item()
            F_sqr_sum += (free_energy**2).sum().item()
            S_sum += entropy.sum().item()
            S_sqr_sum += (entropy**2).sum().item()
            E_sum += energy.sum().item()
            E_sqr_sum += (energy**2).sum().item()

        if args.print_step and (step + 1) % args.print_step == 0:
            count = args.batch_size * (step + 1)
            F_ufloat = get_mean_err(count, F_sum, F_sqr_sum)
            S_ufloat = get_mean_err(count, S_sum, S_sqr_sum)
            E_ufloat = get_mean_err(count, E_sum, E_sqr_sum)
            used_time = time.time() - start_time
            print(
                'count = {}, F = {:.2u}, S = {:.2u}, E = {:.2u}, used_time = {:.3f}'.
                format(count, F_ufloat, S_ufloat, E_ufloat, used_time))
