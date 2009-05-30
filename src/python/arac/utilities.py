#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

"""This module contains several useful utility and convenience functions for
everyday work with arac.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import operator

import scipy


def block_permutation(shape, blockshape):
    # TODO: move tests to testsuite
    """
    >>> block_permutation(shape=(4, 4), blockshape=(2, 2))
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]


    >>> block_permutation(shape=(3, 4), blockshape=(1, 2))
    [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]


    >>> block_permutation(shape=(4, 4, 2), blockshape=(2, 2, 1))
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22, 23, 24, 25, 28, 29, 26, 27, 30, 31]


    >>> block_permutation(shape=(4, 4, 2), blockshape=(2, 2, 2))
    [0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23, 8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31]
    """
    # TODO: Sanity checks.
    dim = len(shape)
    product = lambda l: reduce(operator.mul, l, 1)
    dims = []
    for n in range(dim)[::-1]:
        maxindex = shape[n] / blockshape[n]
        chunklength = blockshape[n] * product(shape[:n])
        n_chunks = product(shape) / chunklength
        this_coords = []
        for i in range(n_chunks):
            for j in range(chunklength):
                this_coords.append(i % maxindex)
        dims.append(this_coords)
    coords = zip(*dims)
    return sorted(range(product(shape)), key=lambda x: coords[x])


def params_by_network(network):
  """Return a list of arrays containing the parameters of a network."""
  # FIXME: make clear whether this is a copy or not
  params = []
  for par in network.parametrizeds():
    params.append(par.get_parameters())
  for net in network.networks():
    for par in net.parametrizeds():
      params.append(par.get_parameters())
  return params


def num_params(network):
    return sum(scipy.size(p) for p in params_by_network(network))


def samples_and_targets(dataset):
    return [(dataset.sample(i), dataset.target(i)) for i in
            range(dataset.size())]


def fill_params(net, params):
    params_ = params_by_network(net)
    start = 0
    for p in params_:
        stop = scipy.size(p) + start
        p[:] = params[start:stop]
        start = stop


if __name__ == '__main__':
    import doctest
    doctest.testmod()
