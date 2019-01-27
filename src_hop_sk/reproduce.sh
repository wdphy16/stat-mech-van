#!/bin/sh

# Fig. 3
python3 main.py --ham hop --n 100 --beta 0.3 --net_depth 1 --net_width 1 --z2 --batch_size 10000 --beta_anneal_to 1.5 --beta_inc 0.1 --clip_grad 1 --cuda 0

# Fig. 4(a)
python3 main.py --ham sk --n 20 --beta 0.1 --net_depth 1 --net_width 1 --z2 --batch_size 10000 --beta_anneal_to 5 --beta_inc 0.1 --clip_grad 1 --cuda 0

# Fig. 4(b)
for beta in `seq 0.1 0.1 2`; do
    python3 main_inv_ising.py --n 20 --beta $beta --net_depth 2 --net_width 5 --z2 --batch_size 10000 --clip_grad 1 --cuda 0
done
