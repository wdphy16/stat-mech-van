#!/usr/bin/env python3
#
# Solving Statistical Mechanics using Variational Autoregressive Networks
# Hopfield model and SK model

import time

import numpy as np
import torch
from torch import nn

from args import args
from made2 import MADE
from hop import HopfieldModel
from sk import SKModel
from utils import (
    ensure_dir,
    init_out_dir,
    my_log,
    print_args,
)


def main():
    start_time = time.time()

    init_out_dir()
    print_args()

    if args.ham == 'hop':
        ham = HopfieldModel(args.n, args.beta, args.device, seed=args.seed)
    elif args.ham == 'sk':
        ham = SKModel(args.n, args.beta, args.device, seed=args.seed)
    else:
        raise ValueError('Unknown ham: {}'.format(args.ham))
    ham.J.requires_grad = False

    net = MADE(**vars(args))
    net.to(args.device)
    my_log('{}\n'.format(net))

    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    my_log('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    if args.beta_anneal_to < args.beta:
        args.beta_anneal_to = args.beta
    beta = args.beta
    while beta <= args.beta_anneal_to:
        for step in range(args.max_step):
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
                energy = ham.energy(sample)
                loss = log_prob + beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()

            if args.clip_grad > 0:
                # nn.utils.clip_grad_norm_(params, args.clip_grad)
                parameters = list(filter(lambda p: p.grad is not None, params))
                max_norm = float(args.clip_grad)
                norm_type = 2
                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item()**norm_type
                    total_norm = total_norm**(1 / norm_type)
                    clip_coef = max_norm / (total_norm + args.epsilon)
                    for p in parameters:
                        p.grad.data.mul_(clip_coef)

            optimizer.step()

            train_time += time.time() - train_start_time

            if args.print_step and step % args.print_step == 0:
                free_energy_mean = loss.mean() / beta / args.n
                free_energy_std = loss.std() / beta / args.n
                entropy_mean = -log_prob.mean() / args.n
                energy_mean = energy.mean() / args.n
                mag = sample.mean(dim=0)
                mag_mean = mag.mean()
                if step > 0:
                    sample_time /= args.print_step
                    train_time /= args.print_step
                used_time = time.time() - start_time
                my_log(
                    'beta = {:.3g}, # {}, F = {:.8g}, F_std = {:.8g}, S = {:.5g}, E = {:.5g}, M = {:.5g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                    .format(
                        beta,
                        step,
                        free_energy_mean.item(),
                        free_energy_std.item(),
                        entropy_mean.item(),
                        energy_mean.item(),
                        mag_mean.item(),
                        sample_time,
                        train_time,
                        used_time,
                    ))
                sample_time = 0
                train_time = 0

        with open(args.fname, 'a', newline='\n') as f:
            f.write('{} {} {:.3g} {:.8g} {:.8g} {:.8g} {:.8g}\n'.format(
                args.n,
                args.seed,
                beta,
                free_energy_mean.item(),
                free_energy_std.item(),
                energy_mean.item(),
                entropy_mean.item(),
            ))

        if args.ham == 'hop':
            ensure_dir(args.out_filename + '_sample/')
            np.savetxt(
                '{}_sample/sample{:.2f}.txt'.format(args.out_filename, beta),
                sample.cpu().numpy(),
                delimiter=' ',
                fmt='%d')
            np.savetxt(
                '{}_sample/log_prob{:.2f}.txt'.format(args.out_filename, beta),
                log_prob.cpu().detach().numpy(),
                delimiter=' ',
                fmt='%.5f')

        beta += args.beta_inc


if __name__ == '__main__':
    main()
