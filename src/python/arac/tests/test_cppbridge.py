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
        net = arac.cppbridge.Network()
        net.add_module(l1, arac.cppbridge.Network.InputModule)
        net.add_module(l2, arac.cppbridge.Network.OutputModule)
        net.add_connection(con)
        inpt = scipy.array((3., 4.))
        result = scipy.zeros(2)
        net.activate(inpt, result)
        self.assertArrayNear(result, scipy.array((11, 25)))

    def testNetworkClear(self):
        net = arac.cppbridge.Network()
        l = arac.cppbridge.LinearLayer(1)
        net.add_module(l, arac.cppbridge.Network.InputOutputModule)
        inputbuffer = scipy.ones((1, 1))
        outputbuffer = scipy.ones((1, 1))
        inputerror = scipy.ones((1, 1))
        outputerror = scipy.ones((1, 1))
        net.init_input(inputbuffer)
        net.init_output(outputbuffer)
        net.init_inerror(inputerror)
        net.init_outerror(outputerror)
        net.clear()
        self.assert_((inputbuffer == 0).all())
        self.assert_((outputbuffer == 0).all())
        self.assert_((inputerror == 0).all())
        self.assert_((outputerror == 0).all())
        
    def testCreateHugeModule(self):
        s = 1000
        l = arac.cppbridge.LinearLayer(s)
        self.assertEqual(l.insize(), s)
        self.assertEqual(l.outsize(), s)
        l = arac.cppbridge.MdlstmLayer(1, s)
        self.assertEqual(l.insize(), 5 * s)
        self.assertEqual(l.outsize(), 2 * s)
        
        
class TestDatasets(TestCase):

    # TODO: Test multiple data rows.
    # TODO: Test out-of-bounds access.

    def testSupervisedSimpleDataset(self):
        ds = arac.cppbridge.SupervisedSimpleDataset(1, 1)
        sample = scipy.random.random(1)
        target = scipy.random.random(1)
        ds.append(sample, target)
        self.assertEqual(1, ds.size())
        self.assertArrayEqual(ds.sample(0), sample)
        self.assertArrayEqual(ds.target(0), target)

    def testSupervisedSemiSequentialDataset(self):
        ds = arac.cppbridge.SupervisedSemiSequentialDataset(1, 1)
        sampleseq = scipy.random.random((1, 2))
        target = scipy.random.random(1)
        ds.append(sampleseq, target)
        self.assertEqual(1, ds.size())
        self.assertArrayEqual(ds.sample(0), sampleseq)
        self.assertArrayEqual(ds.target(0), target)
        
    def testSupervisedSequentialDataset(self):
        ds = arac.cppbridge.SupervisedSequentialDataset(1, 1)
        sampleseq = scipy.random.random((1, 2))
        targetseq = scipy.random.random((1, 2))
        ds.append(sampleseq, targetseq)
        self.assertEqual(1, ds.size())
        self.assertArrayEqual(ds.sample(0), sampleseq)
        self.assertArrayEqual(ds.target(0), targetseq)


if __name__ == "__main__":
    unittest.main()  