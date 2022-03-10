# -*- coding: utf-8 -*-
# file: information.py.py
# time: 01/03/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from sklearn.metrics.cluster import normalized_mutual_info_score
# a = [0, 0, 1, 1]
# b = [0, 0, 1, 1]
# a = [random.randint(0,1) for _ in range(10)]
# b = [random.randint(0,1) for _ in range(10)]
X=[1,1,1,1,1,0,0,0,0,0]
Y=[1,0,0,1,1,0,0,0,1,0]
X = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
Y = [1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0]
NMI = normalized_mutual_info_score(X, Y)
print('Mutual information:', round(NMI,3))

# example of calculating cross entropy for identical distributions
from math import log2

# calculate cross entropy
import numpy as np


def cross_entropy(p, q):
    p = np.array(p)+1e-15
    q = np.array(q)+1e-15

    return -sum([p[i]*log2(q[i]) for i in range(len(p))])

# define data
# p = [1, 0, 0]
# q = [0.9, 0.07,  0.03]
# p = [0, 1, 0]
# q = [0.83, 0.12,  0.05]
p = [0, 0, 1]
q = [0.1, 0.2,  0.7]
# calculate cross entropy H(P, P)
ce_pp = cross_entropy(p, p)
ce_pq = cross_entropy(p, q)

print('H(P, P): %.3f bits' % ce_pp, 'H(P, Q): %.3f bits' % ce_pq)


# calculate cross entropy for classification problem
from math import log
from numpy import mean

# calculate cross entropy
def cross_entropy(p, q):
    return -sum([p[i]*log2(q[i]) for i in range(len(p))])

# define classification data
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# calculate cross entropy for each example
results = list()
for i in range(len(p)):
    # create the distribution for each event {0, 1}
    expected = [1.0 - p[i], p[i]]
    predicted = [1.0 - q[i], q[i]]
    # calculate cross entropy for the two events
    ce = cross_entropy(expected, predicted)
    print('>[y=%.1f, yhat=%.1f] ce: %.3f bits' % (p[i], q[i], ce))
    results.append(ce)

# calculate the average cross entropy
mean_ce = mean(results)
print('Average Cross Entropy: %.3f bits' % mean_ce)