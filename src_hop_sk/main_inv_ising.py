#!/usr/bin/env python3
#
# Solving Statistical Mechanics using Variational Autoregressive Networks
# Inverse Ising problem

import time
import pickle

import numpy as np
import torch

from args import args
from made2 import MADE
from sk import SKModel
from utils import (
    default_dtype_torch,
    init_out_dir,
    my_log,
    print_args,
)


def main():
    start_time = time.time()

    init_out_dir()
    print_args()

    fsave_name = 'n{}b{:.2f}D{}.pickle'.format(args.n, args.beta, args.seed)
    with open(fsave_name, 'rb') as f:
        sk_true = pickle.load(f)

    sk_true.J = sk_true.J.to(args.device)
    assert args.n == sk_true.n
    assert args.beta == sk_true.beta

    sk = SKModel(args.n, args.beta, args.device, seed=args.seed)
    my_log('diff = {:.8g}'.format(sk_true.J_diff(sk.J)))

    C_data = sk_true.C_model.to(args.device, default_dtype_torch)
    C = C_data
    C_inv = torch.inverse(C)

    J_nmf = (torch.eye(args.n).to(C.device) - C_inv) / args.beta
    idx = range(args.n)
    J_nmf[idx, idx] = 0
    diff_nmf = sk_true.J_diff(J_nmf.to(args.device))
    my_log('diff_NMF = {:.8g}'.format(diff_nmf))

    J_IP = 1 / 2 * torch.log((1 + C) / (1 - C + args.epsilon)) / args.beta
    idx = range(args.n)
    J_IP[idx, idx] = 0
    diff_IP = sk_true.J_diff(J_IP.to(args.device))
    my_log('diff_IP = {:.8g}'.format(diff_IP))

    J_SM = -C_inv + J_IP * args.beta - C / (1 - C**2 + args.epsilon)
    J_SM /= args.beta
    idx = range(args.n)
    J_SM[idx, idx] = 0
    diff_SM = sk_true.J_diff(J_SM.to(args.device))
    my_log('diff_SM = {:.8g}'.format(diff_SM))

    sk.J.data[:] = 0

    net = MADE(**vars(args))
    net.to(args.device)
    my_log('{}\n'.format(net))

    params = list(net.parameters())
    params = [sk.J]
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    params2 = [sk.J]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    optimizer2 = torch.optim.Adam(params2, lr=0.01)

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    my_log('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    diff = 10000
    for step2 in range(5000):
        optimizer2.zero_grad()
        for step in range(args.max_step + 1):
            optimizer.zero_grad()

            sample_start_time = time.time()
            with torch.no_grad():
                sample, x_hat = net.sample(args.batch_size)
            assert not sample.requires_grad
            assert not x_hat.requires_grad
            sample_time += time.time() - sample_start_time

            train_start_time = time.time()

            log_prob = net.log_prob(sample)
            with torch.no_grad():
                energy = sk.energy(sample)
                loss = log_prob + args.beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()

            optimizer.step()

            train_time += time.time() - train_start_time

        with torch.no_grad():
            sample, x_hat = net.sample(10**6)
        assert not sample.requires_grad
        assert not x_hat.requires_grad

        m_model = torch.mean(sample, 0).view(sample.shape[1], 1)
        C_model = sample.t() @ sample / sample.shape[0] - m_model @ m_model.t()

        C_diff = C_data - C_model
        C_norm = C_diff.norm(2)

        sk.J.grad = -C_diff / (C_norm + args.epsilon)
        optimizer2.step()

        diff_old = diff
        diff = sk_true.J_diff(sk.J)
        my_log('# {}, diff_J = {}, diff_C = {}'.format(
            step2, diff, torch.sqrt(torch.mean(C_diff**2))))

        idx = range(args.n)
        sk.J.data[idx, idx] = 0

    with open(args.fname, 'a', newline='\n') as f:
        f.write('{} {} {:.3g} {:.8g}\n'.format(args.n, args.seed, args.beta,
                                               diff))


if __name__ == '__main__':
    main()
