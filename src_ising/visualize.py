#!/usr/bin/env python3
#
# Visualize filters in the network

import time
from math import sqrt

import torch
import torchvision
from torch import nn

from args import args
from made import MADE
from pixelcnn import PixelCNN
from utils import ensure_dir, get_ham_args_features

# Set args here or through CLI to match the state
args.ham = 'afm'
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

args.max_step = 10**4
args.print_step = 100
args.visual_step = 1000
state_dir = 'out'

ham_args, features = get_ham_args_features()
state_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}_save/10000.state'.format(
    **locals())

target_layer = 1
num_channel = 1
out_dir = '../support/fig/filters/{ham_args}/{features}/layer{target_layer}'.format(
    **locals())

if __name__ == '__main__':
    ensure_dir(out_dir + '/')

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
    net.load_state_dict(state['net'])

    sample = torch.zeros([num_channel, 1, args.L, args.L], requires_grad=True)
    nn.init.normal_(sample)

    optimizer = torch.optim.Adam([sample], lr=1e-3, weight_decay=1)

    start_time = time.time()
    for step in range(args.max_step + 1):
        optimizer.zero_grad()

        x = sample
        for idx, layer in enumerate(net.net):
            x = layer(x)
            if idx == target_layer:
                break

        sample
        loss = torch.mean(
            torch.stack([
                torch.mean(x[channel, channel])
                for channel in range(num_channel)
            ]))
        loss.backward()

        optimizer.step()

        if args.print_step and step % args.print_step == 0:
            used_time = time.time() - start_time
            print('step = {}, loss = {:.8g}, used_time = {:.3f}'.format(
                step, loss, used_time))

        if args.visual_step and step % args.visual_step == 0:
            torchvision.utils.save_image(
                sample,
                '{}/{}.png'.format(out_dir, step),
                nrow=int(sqrt(sample.shape[0])),
                padding=2,
                normalize=True)
