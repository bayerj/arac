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
        
    def testParametrized(self):
        l1 = arac.cppbridge.LinearLayer(2)
        l2 = arac.cppbridge.LinearLayer(2)
        c = arac.cppbridge.FullConnection(l1, l2)
        params = scipy.array((1., 2., 3., 4.))
        c.set_parameters(params)
        self.assertEqual(c.get_parameters().ctypes.data, params.ctypes.data)
        derivs = scipy.array((-1., -2., -3., -4.))
        c.set_derivatives(derivs)
        self.assertEqual(c.get_derivatives().ctypes.data, derivs.ctypes.data)
        
    def testNetworkParametrizeds(self):
        l1 = arac.cppbridge.LinearLayer(1)
        l2 = arac.cppbridge.LinearLayer(1)
        l3 = arac.cppbridge.LinearLayer(1)
        c1 = arac.cppbridge.FullConnection(l1, l2)
        c2 = arac.cppbridge.FullConnection(l2, l3)
        net = arac.cppbridge.Network()
        net.add_module(l1)
        net.add_module(l2)
        net.add_module(l3)
        net.add_connection(c1)
        net.add_connection(c2)
        paras = net.parametrizeds()
        pointer = lambda p: p.get_parameters().ctypes.data
        self.assertEqual(pointer(paras[0]), pointer(c1))
        self.assertEqual(pointer(paras[1]), pointer(c2))
        
    def testConvolveConnection(self):
        l1 = arac.cppbridge.LinearLayer(3)
        l2 = arac.cppbridge.LinearLayer(15)
        c = arac.cppbridge.ConvolveConnection(l1, l2, 1, 5)
        params = scipy.array((1, 2, 3, 4, 5), dtype='float64')
        derivs = scipy.zeros(5)
        c.set_parameters(params)
        c.set_derivatives(derivs)
        net = arac.cppbridge.Network()
        net.add_module(l1, arac.cppbridge.Network.InputModule)
        net.add_module(l2, arac.cppbridge.Network.OutputModule)
        net.add_connection(c)
        
        inpt = scipy.array((2, 3, 6), dtype='float64')
        result = scipy.zeros(15)
        net.activate(inpt, result)
        solution = scipy.array((2., 4., 6., 8., 10., 
                                3., 6., 9., 12., 15.,
                                6., 12., 18., 24., 30.))
        self.assertArrayNear(result, solution)
        
        error = scipy.array([(-i - 1) for i in range(15)], dtype='float64')
        inerror = scipy.zeros(3)
        net.back_activate(error, inerror)
        solution = scipy.array((-55, -130, -205), dtype='float64')
        self.assertArrayNear(inerror, solution)
        
        solution = scipy.array((-86.,  -97., -108., -119., -130.))
        self.assertArrayNear(derivs, solution)
        
    def testBlockPermutationConnection(self):
        l1 = arac.cppbridge.LinearLayer(16)
        l2 = arac.cppbridge.LinearLayer(16)
        c1 = arac.cppbridge.BlockPermutationConnection(l1, l2, (4, 4), (2, 2))
        # c1 = arac.cppbridge.IdentityConnection(l1, l2)
        net = arac.cppbridge.Network()
        net.add_module(l1, arac.cppbridge.Network.InputModule)
        net.add_module(l2, arac.cppbridge.Network.OutputModule)
        net.add_connection(c1)
        result = scipy.zeros(16)
        inpt = scipy.array([float(i) for i in range(16)])
        net.activate(inpt, result)
        solution = scipy.array(
            (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15), 
            dtype='float64')
        self.assertArrayNear(solution, result)

    def testMdrnn(self):
        net = arac.cppbridge.LinearMdrnn(2, 1)
        net.set_sequence_shape(0, 2)
        net.set_sequence_shape(1, 2)
        net.sort()
        
        feedcon, con1, con2, biascon = net.parametrizeds()
        feedpars = feedcon.get_parameters()
        pars1 = con1.get_parameters()
        pars2 = con2.get_parameters()
        feedpars[0] = 1
        pars1[0] = 0.5
        pars2[0] = 2.0
        
        result = scipy.zeros(4)
        inpt = scipy.array([1., 2., 3., 4.])
        net.activate(inpt, result)
        self.assertArrayNear(result, scipy.array((1., 2.5, 5., 11.5)))
        
        error = scipy.array((-1., -2., -3., -4.))
        error = scipy.array((2., 4., 8., 10.))
        inerror = scipy.zeros(4)
        net.back_activate(error, inerror)
        self.assertArrayNear(inerror, scipy.array((40., 24., 13., 10.)))

        
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
        
        
class TestOptimizers(TestCase):
    
    # TODO: Make the tests somehow reasonable. Here, it is only tested that the
    # optimizers don't crash and that the API does not throw any errors.
    
    def testSimpleBackprop(self):
        ds = arac.cppbridge.SupervisedSimpleDataset(1, 1)
        # Add some data points to the dataset.
        for _ in xrange(5):
            ds.append(scipy.random.random(1), scipy.random.random(1))

        l1 = arac.cppbridge.LinearLayer(1)
        l2 = arac.cppbridge.LinearLayer(1)
        net = arac.cppbridge.Network()
        net.add_module(l1, arac.cppbridge.Network.InputModule)
        net.add_module(l2, arac.cppbridge.Network.OutputModule)
        con = arac.cppbridge.FullConnection(l1, l2)
        net.add_connection(con)
        optimizer = arac.cppbridge.SimpleBackprop(net, ds)
        for _ in xrange(10):
            optimizer.train_stochastic()
        
        
if __name__ == "__main__":
    unittest.main()  
