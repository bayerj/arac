#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittests for the arac.pybrainbridge module."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import unittest
import scipy

import arac.pybrainbridge as pybrainbridge

from arac.tests.common import TestCase

from pybrain.structure import (
    LinearLayer, 
    BiasUnit,
    SigmoidLayer, 
    TanhLayer,
    LSTMLayer,
    SoftmaxLayer,
    PartialSoftmaxLayer,
    IdentityConnection, 
    FullConnection,
    Network,
    RecurrentNetwork,
    FeedForwardNetwork
)


scipy.random.seed(0)


class TestNetworkEquivalence(TestCase):
    
    def two_layer_network(self, net):
        inlayer = SigmoidLayer(2, 'in')
        outlayer = LinearLayer(2, 'out')
        con = FullConnection(inlayer, outlayer)
        con.params[:] = 1, 2, 3, 4
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)
        net.sortModules()

    def rec_two_layer_network(self, net):
        inlayer = LinearLayer(2, 'in')
        outlayer = LinearLayer(2, 'out')
        rcon = IdentityConnection(inlayer, outlayer)
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addRecurrentConnection(rcon)
        net.sortModules()

    def lstm_network(self, net):
        i = LinearLayer(1)
        h = LSTMLayer(1)
        o = LinearLayer(1)
        b = BiasUnit()
        net.addModule(b)
        net.addOutputModule(o)
        net.addInputModule(i)
        net.addModule(h)
        net.addConnection(FullConnection(i, h))
        net.addConnection(FullConnection(b, h))
        net.addRecurrentConnection(FullConnection(h, h))
        net.addConnection(FullConnection(h, o))
        net.sortModules()
        
    def equivalence_feed_forward(self, builder):
        _net = pybrainbridge._FeedForwardNetwork()
        builder(_net)
        net = FeedForwardNetwork()
        builder(net)
        
        inpt = scipy.random.random(net.indim)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)
        self.assertArrayNear(pybrain_res, arac_res)

        error = scipy.random.random(net.outdim)
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)
                             
        inpt = scipy.random.random(net.indim)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)
        self.assertArrayNear(pybrain_res, arac_res)
                             
        error = scipy.random.random(net.outdim)
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)
                          
    def equivalence_recurrent(self, builder):
        _net = pybrainbridge._RecurrentNetwork()
        builder(_net)
        net = RecurrentNetwork()
        builder(net)
        
        self.assertEqual(_net.proxies[_net].get_mode(), 2, 
                         "Mode of _RecurrentNetwork is not 'Sequential'.")
        self.assertEqual(_net.proxies[_net['in']].get_mode(), 2, 
                         "Mode of input layer is not 'Sequential'.")
        self.assertEqual(_net.proxies[_net['out']].get_mode(), 2, 
                         "Mode of output layer is not 'Sequential'.")

        self.assertEqual(net.offset, 0)
        self.assertEqual(_net.offset, 0)
        self.assertEqual(_net.proxies[_net['in']].timestep(), 0)
        self.assertEqual(_net.proxies[_net['out']].timestep(), 0)
        
        inpt = range(net.indim)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)
        
        print _net.inputbuffer
        print _net.outputbuffer
        
        self.assertArrayNear(pybrain_res, arac_res)
        self.assertEqual(net.offset, 1)
        self.assertEqual(_net.offset, 1)
        self.assertEqual(_net.proxies[_net['in']].timestep(), 1)
        self.assertEqual(_net.proxies[_net['out']].timestep(), 1)

        inpt = range(2, net.indim + 2)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)

        print "Network input", _net.inputbuffer
        print "Network output", _net.outputbuffer
        print "Inlayer input", _net['in'].inputbuffer
        print "Inlayer output", _net['in'].outputbuffer
        print "Outlayer input", _net['out'].inputbuffer
        print "Outlayer output", _net['out'].outputbuffer

        self.assertEqual(net.offset, 2)
        self.assertEqual(_net.offset, 2)
        self.assertArrayNear(pybrain_res, arac_res)
        self.assertEqual(_net.proxies[_net['in']].timestep(), 2)
        self.assertEqual(_net.proxies[_net['out']].timestep(), 2)

        error = range(net.outdim)
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)
                             
        error = range(net.outdim)[::-1]
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)

    def _testTwoLayerNetwork(self):
        self.equivalence_feed_forward(self.two_layer_network)

    def testRecTwoLayerNetwork(self):
        self.equivalence_recurrent(self.rec_two_layer_network)

    def _testLstmNetwork(self):
        self.equivalence_recurrent(self.lstm_network)


if __name__ == "__main__":
    unittest.main()  