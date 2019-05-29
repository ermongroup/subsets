from collections import OrderedDict, defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import click


EPSILON = np.finfo(np.float32).tiny

curr_dir = Path(__file__).expanduser().resolve().parent


def run_subset_sampling(fn, possible_subsets, p, k, num_samples):
    pair_count = defaultdict(lambda: 0)
    for i in range(num_samples):
        subset, count_weight = fn(p, k)
        pair_count[tuple(subset)] += count_weight
    pair_probs = OrderedDict((k,  pair_count[k] / num_samples)
                             for k in possible_subsets)
    return pair_probs


def sample_subset_simple(p, k):
    n = len(p)
    z = np.random.gumbel(size=n)
    keys = np.log(p) + z
    top_k_idxs = np.sort(np.argsort(keys)[-k:])
    return top_k_idxs, 1


def sample_subset_continuous(p, k, t):
    n = len(p)
    z = np.random.gumbel(size=n)
    w = np.log(p)
    keys = w + z
    onehot_approx = np.zeros_like(p)
    khot_list = []
    for i in range(k):
        khot_mask = np.maximum(1 - onehot_approx, EPSILON)
        keys += np.log(khot_mask)
        onehot_approx = softmax(keys / t)
        khot_list.append(onehot_approx)
    return np.sum(np.asarray(khot_list), 0)


def softmax(logits):
    exp_logits = np.exp(logits - np.amax(logits))
    return exp_logits / np.sum(exp_logits)


def histogram_test(k, n, plot, num_samples, t=None):
    p = np.arange(n) + 1
    p = p / np.sum(p)
    print(f"Item weights: {p}")
    n = len(p)
    if not t:
        t = [10, 1, 0.1]

    # calculate theoretical probabilities
    probs_theoretical = OrderedDict()
    for subset in combinations(np.arange(n), k):
        subset_probs = p[list(subset)]
        subset_prob = 0
        for permutation in permutations(subset_probs):
            curr_prob = 1.0
            denom = np.sum(p)
            for i in range(len(permutation)):
                curr_prob *= permutation[i] / denom
                denom -= permutation[i]
            subset_prob += curr_prob
        probs_theoretical[tuple(subset)] = subset_prob

    # run simple
    probs_simple = run_subset_sampling(sample_subset_simple,
                                            probs_theoretical.keys(),
                                            p, k, num_samples)

    # run continuous
    data_dict = {}
    def run_continuous_sampling(temp):
        def sample_subset_continuous_partial(p, k):
            onehot_sum_np = sample_subset_continuous(p, k, temp)
            onehot_sum_np = np.squeeze(onehot_sum_np)
            # return the hard max of onehot_sum as the choices
            top_k_idxs = np.sort(np.argsort(onehot_sum_np)[-k:])
            # return the min of the top k as the count weight
            return top_k_idxs, 1

        return run_subset_sampling(sample_subset_continuous_partial,
                                   probs_theoretical.keys(),
                                   p, k, num_samples)

    curr_data_dict = OrderedDict([
        (f'T=' + str(curr_t),
         list(run_continuous_sampling(curr_t).values()))
        for curr_t in t])
    data_dict.update(curr_data_dict)

    # data_dict['simple'] = list(probs_simple.values())
    data_dict['theoretical'] = list(probs_theoretical.values())

    # need to add 1 for display since indexing of elements starts at 1
    xlabels = [str(tuple(np.asarray(k)+1)) for k in probs_theoretical.keys()]
    df = pd.DataFrame(data_dict, index=xlabels)

    # get TV distances
    for col in df.columns:
        if col != 'theoretical':
            tv = 0.5 * np.sum(np.abs(df[col] - df['theoretical']))
            print(f"TV distance of {col} to theoretical: {tv}")

    if plot:
        fig, ax = plt.subplots()
        ci = [1.96*np.sqrt(p*(1-p)/num_samples) for p in probs_theoretical.values()]
        for i, col in enumerate(df.columns):
            xs = (0.2*(df.shape[1]+1))*np.arange(df.shape[0]) + 0.2*i
            ys = df[col]
            if col =='theoretical':
                plt.bar(xs, height=ys, width=0.2, tick_label=None, label=col)
            else:
                plt.bar(xs, height=ys, width=0.2, yerr=ci, tick_label=None, label=col)
        ax.set_xticks((0.2*(df.shape[1]+1))*np.arange(df.shape[0]) + 0.2*df.shape[1]/2)
        ax.set_xticklabels(xlabels)
        plt.legend()
        plt.xlabel('Subset Elements')
        plt.ylabel('Probability')
        plt.title('Empirical Subset Distributions')
        plt.show()
    return data_dict


@click.command()
@click.option('--k', default=2, required=True, help='Size of subset', type=int)
@click.option('--n', default=4, required=True, help='Total number of elements', type=int)
@click.option('--plot', default=False, is_flag=True, help='Do plotting')
def main(k=2, n=4, plot=True, tvtest=False):
    histogram_test(k, n, plot, 10000)


if __name__ == '__main__':
    main()
