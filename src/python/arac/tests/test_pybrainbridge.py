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
        con = IdentityConnection(inlayer, outlayer)
        rcon = IdentityConnection(inlayer, outlayer)
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)
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
        net.params[:] = range(1, 14)
        
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
        
        inpt = range(net.indim)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)
        
        self.assertArrayNear(pybrain_res, arac_res)

        inpt = range(2, net.indim + 2)
        pybrain_res = net.activate(inpt)
        arac_res = _net.activate(inpt)

        self.assertArrayNear(pybrain_res, arac_res)

        error = range(net.outdim)
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)
                             
        error = range(net.outdim)[::-1]
        pybrain_res = net.backActivate(error)
        arac_res = _net.backActivate(error)
        self.assertArrayNear(pybrain_res, arac_res)

    def testTwoLayerNetwork(self):
        self.equivalence_feed_forward(self.two_layer_network)

    def _testRecTwoLayerNetwork(self):
        self.equivalence_recurrent(self.rec_two_layer_network)
        
    def _testTimesteps(self):
        _net = pybrainbridge._RecurrentNetwork()
        self.rec_two_layer_network(_net)
        
        netproxy = _net.proxies[_net]
        inproxy = _net.proxies[_net['in']]
        outproxy = _net.proxies[_net['out']]
        conproxy = _net.proxies[_net.connections[_net['in']][0]]
        rconproxy = _net.proxies[_net.recurrentConns[0]]
        
        proxies = netproxy, inproxy, outproxy, conproxy, rconproxy
        for proxy in proxies:
            self.assertEqual(proxy.get_mode(), 2)
            
        self.assertEqual(_net.offset, 0)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 0,
                             "%s has wrong timestep." % proxy)

        _net.activate((0, 0))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 1)

        _net.activate((0, 0))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 2)

        _net.activate((0, 0))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 3)

        _net.backActivate((0, 0))
        self.assertEqual(_net.offset, 2)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 2)

        _net.backActivate((0, 0))
        self.assertEqual(_net.offset, 1)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 1)

        _net.backActivate((0, 0))
        self.assertEqual(_net.offset, 0)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 0)

    def testLstmNetwork(self):
        self.equivalence_recurrent(self.lstm_network)


if __name__ == "__main__":
    unittest.main()  