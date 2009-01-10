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

        self.assertEqual(c.get_incomingstart(), 0)
        self.assertEqual(c.get_incomingstop(), 2)
        self.assertEqual(c.get_outgoingstart(), 0)
        self.assertEqual(c.get_outgoingstop(), 2)

        self.assert_(c.this is not None)
        
    def testNetwork(self):
        l1 = arac.cppbridge.LinearLayer(2)
        l2 = arac.cppbridge.LinearLayer(2)
        
        params = scipy.array((1, 2, 3, 4), dtype='float64')
        derivs = scipy.zeros(4)
        con = arac.cppbridge.FullConnection(l1, l2, params, derivs, 0, 2, 0, 2)
        # con = arac.cppbridge.IdentityConnection(l1, l2, 0, 2, 0, 2)
        
        net = arac.cppbridge.Network()
        net.add_module(l1, arac.cppbridge.Network.InputModule)
        net.add_module(l2, arac.cppbridge.Network.OutputModule)
        net.add_connection(con)
        inpt = scipy.array((3., 4.))
        result = scipy.zeros(2)
        net.activate(inpt, result)
        self.assertArrayNear(result, scipy.array((9, 28)))


if __name__ == "__main__":
    unittest.main()  