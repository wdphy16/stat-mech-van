#!/bin/sh

# Fig. 2(a)
for beta in `seq 0.1 0.1 1`; do
    python3 main.py --ham fm --lattice sqr --L 16 --beta $beta --net pixelcnn --net_depth 3 --net_width 64 --half_kernel_size 6 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0
    python3 main.py --ham fm --lattice sqr --L 16 --beta $beta --net pixelcnn --net_depth 6 --net_width 64 --half_kernel_size 3 --bias --z2 --res_block --beta_anneal 0.998 --clip_grad 1 --cuda 0
    python3 main.py --ham fm --lattice sqr --L 16 --beta $beta --net made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0
    python3 main.py --ham fm --lattice sqr --L 16 --beta $beta --net made --net_depth 6 --net_width 2 --bias --z2 --res_block --beta_anneal 0.998 --clip_grad 1 --cuda 0
done

# Fig. 2(b)
for L in `seq 4 2 16`; do
    for beta in `seq 1 1 5`; do
        python3 main.py --ham afm --lattice tri --L $L --beta $beta --net pixelcnn --net_depth 3 --net_width 64 --half_kernel_size 6 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0
    done
done
