#!/usr/bin/env python3
#
# Sample from a trained network

import re
import time
from collections import namedtuple
from math import sqrt

import numpy as np
import torch
from uncertainties import ufloat

import ising
from args import args
from made import MADE
from pixelcnn import PixelCNN
from utils import get_ham_args_features, ignore_param, print_args

default_dtype = np.float64

# Set args here or through CLI to match the state
args.ham = 'afm'
args.lattice = 'tri'
args.boundary = 'periodic'
args.L = 4
args.beta = 1

args.net = 'pixelcnn'
args.net_depth = 3
args.net_width = 64
args.half_kernel_size = 1
args.bias = False
args.z2 = False
args.res_block = False
args.beta_anneal = 0
args.clip_grad = 0

args.batch_size = 10**3
args.max_step = 10
args.print_step = 1
state_dir = 'out'

ham_args, features = get_ham_args_features()
state_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}_save/{{}}.state'.format(
    **locals())
log_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}.log'.format(
    **locals())
final_window = 100

Run = namedtuple('Run', [
    'step', 'F_arr', 'F_std_arr', 'S_arr', 'E_arr', 'param_count', 'used_time'
])


def read_log(filename):
    step_arr = []
    F_arr = []
    F_std_arr = []
    S_arr = []
    E_arr = []
    param_count = 0
    used_time = 0
    with open(filename, 'r') as f:
        for line in f:
            match = re.compile(r'parameters: (.*)').search(line)
            if match:
                param_count = int(match.group(1))
                continue

            match = re.compile(
                r'step = (.*?),.*F = (.*?),.*F_std = (.*?),.*S = (.*?),.*E = (.*?),.*used_time = (.*)'
            ).search(line)
            if match:
                step_arr.append(int(match.group(1)))
                F_arr.append(float(match.group(2)))
                F_std_arr.append(float(match.group(3)))
                S_arr.append(float(match.group(4)))
                E_arr.append(float(match.group(5)))
                used_time = float(match.group(6))

    step_arr = np.array(step_arr, dtype=int)
    F_arr = np.array(F_arr, dtype=default_dtype)
    F_std_arr = np.array(F_std_arr, dtype=default_dtype)
    S_arr = np.array(S_arr, dtype=default_dtype)
    E_arr = np.array(E_arr, dtype=default_dtype)

    idx = np.argsort(step_arr)
    step_arr = step_arr[idx]
    F_arr = F_arr[idx]
    F_std_arr = F_std_arr[idx]
    S_arr = S_arr[idx]
    E_arr = E_arr[idx]

    return Run(step_arr, F_arr, F_std_arr, S_arr, E_arr, param_count,
               used_time)


def get_state_step():
    run = read_log(log_filename)
    F_cumsum = np.cumsum(run.F_arr)
    F_avg = (F_cumsum[final_window:] - F_cumsum[:-final_window]) / final_window
    step = np.argmin(F_avg)
    step = int(round((step + final_window) / args.save_step)) * args.save_step
    return step


def get_mean_err(count, x_sum, x_sqr_sum):
    x_mean = x_sum / count
    x_sqr_mean = x_sqr_sum / count
    x_std = sqrt(abs(x_sqr_mean - x_mean**2))
    x_err = x_std / sqrt(count)
    x_ufloat = ufloat(x_mean, x_err)
    return x_ufloat


if __name__ == '__main__':
    print_args(print_fn=print)

    if args.net == 'made':
        net = MADE(**vars(args))
    elif args.net == 'pixelcnn':
        net = PixelCNN(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
    net.to(args.device)
    print('{}\n'.format(net))

    state_filename = state_filename.format(get_state_step())
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
    M_sum = 0
    M_sqr_sum = 0
    M_quad_sum = 0
    start_time = time.time()
    for step in range(args.max_step):
        with torch.no_grad():
            sample, _ = net.sample(args.batch_size)
            log_prob = net.log_prob(sample)
            energy = ising.energy(sample, args.ham, args.lattice,
                                  args.boundary) / args.L**2
            free_energy = energy + 1 / args.beta * log_prob / args.L**2
            entropy = -log_prob / args.L**2
            mag = sample.mean(dim=[1, 2, 3])

            F_sum += free_energy.sum().item()
            F_sqr_sum += (free_energy**2).sum().item()
            S_sum += entropy.sum().item()
            S_sqr_sum += (entropy**2).sum().item()
            E_sum += energy.sum().item()
            E_sqr_sum += (energy**2).sum().item()
            M_sum += mag.abs().sum().item()
            M_sqr_sum += (mag**2).sum().item()
            M_quad_sum += (mag**4).sum().item()

        if args.print_step and (step + 1) % args.print_step == 0:
            count = args.batch_size * (step + 1)
            F_ufloat = get_mean_err(count, F_sum, F_sqr_sum)
            S_ufloat = get_mean_err(count, S_sum, S_sqr_sum)
            E_ufloat = get_mean_err(count, E_sum, E_sqr_sum)
            Cv = args.beta**2 * (E_sqr_sum / count -
                                 (E_sum / count)**2) * args.L**2
            chi = args.beta * (M_sqr_sum / count -
                               (M_sum / count)**2) * args.L**2
            binder_ratio = 1 / 2 * (3 - (M_quad_sum / count) /
                                    (M_sqr_sum / count)**2)
            used_time = time.time() - start_time
            print(
                'count = {}, F = {:.2u}, S = {:.2u}, E = {:.2u}, Cv = {:.8g}, chi = {:.8g}, g = {:.8g}, used_time = {:.3f}'
                .format(count, F_ufloat, S_ufloat, E_ufloat, Cv, chi,
                        binder_ratio, used_time))
