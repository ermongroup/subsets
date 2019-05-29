########################################
# Adapted from https://github.com/ermongroup/neuralsort/
########################################

import torch
from subsets.knn.pl import PL
from subsets.knn.sorting_operator import SortingOperator, SubsetOperator


class DKNN(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False, num_samples=-1):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = SortingOperator(tau=tau, hard=hard)
        self.num_samples = num_samples

    # query: batch_size x p
    # neighbors: 10k x p
    def forward(self, query, neighbors, tau=1.0):
        diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        norms = l2_norms  # .sqrt() # M x 10k
        scores = -norms

        pl_s = PL(scores, tau, hard=False)
        P_hat = pl_s.sample((self.num_samples,))
        top_k = P_hat[:, :, :self.k, :].sum(2)
        return top_k


class SubsetsDKNN(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False, num_samples=-1):
        super(SubsetsDKNN, self).__init__()
        self.k = k
        self.subset_sample = SubsetOperator(k=k, tau=tau, hard=hard)
        self.num_samples = num_samples

    # query: batch_size x p
    # neighbors: 10k x p
    def forward(self, query, neighbors, tau=1.0):
        diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        norms = l2_norms  # .sqrt() # M x 10k
        scores = -norms

        top_k = self.subset_sample(scores)
        return top_k
