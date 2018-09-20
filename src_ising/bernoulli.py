# Bernoulli mixture model, used for baseline

from math import log

import torch
from torch import nn

from utils import default_dtype_torch


class BernoulliMixture(nn.Module):
    def __init__(self, **kwargs):
        super(BernoulliMixture, self).__init__()
        self.L = kwargs['L']
        self.net_width = kwargs['net_width']
        self.z2 = kwargs['z2']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        self.ber_weight = nn.Parameter(
            torch.rand([self.net_width, self.L, self.L]))
        self.mix_weight = nn.Parameter(torch.zeros([self.net_width]))

    def forward(self, x):
        raise NotImplementedError

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # return sample as dummy of x_hat
    def sample(self, batch_size):
        mix_prob = torch.exp(self.mix_weight)
        choice = torch.multinomial(mix_prob, batch_size, replacement=True)

        ber_prob = torch.sigmoid(self.ber_weight)
        ber_prob = ber_prob[choice, None, :, :]
        sample = torch.bernoulli(ber_prob).to(default_dtype_torch) * 2 - 1

        return sample, sample

    def _log_prob(self, sample):
        ber_prob = torch.sigmoid(self.ber_weight)
        ber_prob = ber_prob[None, :, None, :, :]
        mask = (sample + 1) / 2
        mask = mask[:, None, :, :, :]
        sample_prob = ber_prob * mask + (1 - ber_prob) * (1 - mask)
        log_prob = torch.log(sample_prob + self.epsilon)
        log_prob = log_prob.view(sample.shape[0], self.net_width, -1)
        log_prob = log_prob.sum(dim=2)

        mix_prob = torch.exp(self.mix_weight)
        mix_prob = mix_prob / mix_prob.sum()
        log_prob = torch.log(torch.exp(log_prob) @ mix_prob)

        return log_prob

    def log_prob(self, sample):
        log_prob = self._log_prob(sample)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            log_prob_inv = self._log_prob(sample_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob
