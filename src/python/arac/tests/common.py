#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Common functionality used by unittests."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy


def array_equal(arr1, arr2):
    arr1 = scipy.array(arr1)
    arr2 = scipy.array(arr2)
    return (arr1 == arr2).all()