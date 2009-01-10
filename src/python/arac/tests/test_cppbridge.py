#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittests for the arac.pybrainbridge module."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import unittest
import scipy

import arac.cppbridge
from arac.tests.common import TestCase


class TestStructure(TestCase):
    
    def testFullConnection(self):
        params = scipy.array((1, 2, 3, 4), dtype='float64')
        derivs = scipy.zeros(4, dtype='float64')
        l1 = arac.cppbridge.LinearLayer(2)
        l2 = arac.cppbridge.LinearLayer(2)
        c = arac.cppbridge.FullConnection(l1, l2, params, derivs, 0, 2, 0, 2)
        self.assert_(c.this is not None)


if __name__ == "__main__":
    unittest.main()  