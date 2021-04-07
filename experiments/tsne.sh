#!/bin/sh

set -x

cd ../subsets/tsne/

for out_dim in 2 10 30
do
    python train_tsne.py --train --dataset mnist --tau 0.1 --out_dim ${out_dim} --k 1 --epochs 200 --do_pretrain
    python train_tsne.py --train --dataset mnist --tau 0.1 --out_dim ${out_dim} --k 1 --epochs 200

    python train_tsne.py --train --dataset 20newsgroups --tau 0.1 --out_dim ${out_dim} --k 1 --epochs 200 --do_pretrain
    python train_tsne.py --train --dataset 20newsgroups --tau 0.1 --out_dim ${out_dim} --k 1 --epochs 200
done
