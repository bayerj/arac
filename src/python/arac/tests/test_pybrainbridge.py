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
        inlayer = SigmoidLayer(2)
        outlayer = LinearLayer(2)
        con = FullConnection(inlayer, outlayer)
        con.params[:] = 1, 2, 3, 4
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)
        net.sortModules()

    def rec_two_layer_network(self, net):
        inlayer = SigmoidLayer(2)
        outlayer = LinearLayer(2)
        con = FullConnection(inlayer, outlayer)
        con.params[:] = 1, 2, 3, 4
        rcon = FullConnection(inlayer, outlayer)
        rcon.params[:] = 1, 2, 3, 5
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
        
    def equivalence_feed_forward(self, builder):
        self._equivalence(builder, 
                          pybrainbridge._FeedForwardNetwork, 
                          FeedForwardNetwork)
                          
    def equivalence_recurrent(self, builder):
        self._equivalence(builder,
                          pybrainbridge._RecurrentNetwork,
                          RecurrentNetwork)

    def _equivalence(self, builder, aracclass, pybrainclass):
        _net = aracclass()
        builder(_net)
        net = pybrainclass()
        builder(net)
        
        inpt = scipy.random.random(net.indim)
        self.assertArrayNear(net.activate(inpt), 
                             _net.activate(inpt))

        error = scipy.random.random(net.outdim)
        self.assertArrayNear(net.backActivate(error), 
                             _net.backActivate(error))
                             
        inpt = scipy.random.random(net.indim)
        self.assertArrayNear(net.activate(inpt), 
                             _net.activate(inpt))
                             
        error = scipy.random.random(net.outdim)
        self.assertArrayNear(net.backActivate(error), 
                             _net.backActivate(error))
        
    def testTwoLayerNetwork(self):
        self.equivalence_feed_forward(self.two_layer_network)

    def testRecTwoLayerNetwork(self):
        self.equivalence_recurrent(self.rec_two_layer_network)

    def _testLstmNetwork(self):
        self.equivalence_recurrent(self.lstm_network)


if __name__ == "__main__":
    unittest.main()  