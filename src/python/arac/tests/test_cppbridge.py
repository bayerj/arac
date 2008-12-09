#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittests for the arac.cppbridge module."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import unittest
import scipy

import arac.cppbridge as cppbridge

from arac.tests.common import TestCase


class AracCall(TestCase):
    
    def testCall(self):
        result = cppbridge.arac_call("""return_val = 1;""")
        self.assertEqual(result, 1)


class TestSimpleLayer(TestCase):
    
    configurations = {
        'LinearLayer': {
            'inpt': (2, 3),
            'outpt': (2, 3),
            'outerror': (1, -.3),
            'inerror': (1, -.3)},
        
        'SigmoidLayer': {
            'inpt': (-1, 0.5),
            'outpt': (0.2689414213699951, 0.62245933120185459),
            'outerror': (2, 8),
            'inerror': (0.3932238664829637, 1.880029697612756)},
        
        'TanhLayer': {
            'inpt': (-1, 0.5),
            'outpt': (-0.76159415595576485, 0.46211715726000974),
            'outerror': (2, 8),
            'inerror': (0.83994868322805227, 6.2915818637274192)},

        'SoftmaxLayer': {
            'inpt': (2, 4),
            'outpt': (0.11920292202211756, 0.88079707797788243),
            'outerror': (2, 4),
            'inerror': (2, 4)},
    }

    def run_configuration(self, klass):
        conf = self.configurations[klass]
        inpt = scipy.zeros((1, 2))
        outpt = scipy.zeros((1, 2))
        inerror = scipy.zeros((1, 2))
        outerror = scipy.zeros((1, 2))
        layer = cppbridge.SimpleLayer(klass, 2, inpt, outpt, inerror, outerror)
        inpt[0, :] = conf['inpt']
        layer.forward()
        self.assertArrayEqual(outpt[0], conf['outpt'])
        outerror[0, :] = conf['outerror']
        layer.backward()
        self.assertArrayEqual(inerror[0], conf['inerror'])
    
    def testLinearLayer(self):
        self.run_configuration('LinearLayer')

    def testSigmoidLayer(self):
        self.run_configuration('SigmoidLayer')

    def testTanhLayer(self):
        self.run_configuration('TanhLayer')

    def testSoftmaxLayer(self):
        self.run_configuration('SoftmaxLayer')


class TestConnection(TestCase):
    
    def testIdentityConnection(self):
        inlayer_input = scipy.zeros((1, 2), dtype='float64')
        inlayer_output = scipy.zeros((1, 2), dtype='float64')
        inlayer_inerror = scipy.zeros((1, 2), dtype='float64')
        inlayer_outerror = scipy.zeros((1, 2), dtype='float64')
        outlayer_input = scipy.zeros((1, 2), dtype='float64')
        outlayer_output = scipy.zeros((1, 2), dtype='float64')
        outlayer_inerror = scipy.zeros((1, 2), dtype='float64')
        outlayer_outerror = scipy.zeros((1, 2), dtype='float64')

        inlayer = cppbridge.SimpleLayer(
            'LinearLayer', 2, 
            inlayer_input, inlayer_output, 
            inlayer_inerror, inlayer_outerror)
            
        outlayer = cppbridge.SimpleLayer(
            'LinearLayer', 2,
            outlayer_input, outlayer_output, 
            outlayer_inerror, outlayer_outerror)
            
        con = cppbridge.IdentityConnection(inlayer.address, outlayer.address)
            
        inlayer_input[0, :] = 2., 3.
        
        inlayer.forward()
        self.assertArrayNear(inlayer_output[0], (2, 3))
        
        con.forward()
        self.assertArrayNear(outlayer_input[0], (2, 3))

        outlayer.forward()
        self.assertArrayNear(outlayer_output[0], (2, 3))

        outlayer_outerror[0, :] = 0.5, 1.2
        
        outlayer.backward()
        con.backward()
        inlayer.backward()
        
        self.assertArrayNear(inlayer_inerror[0], (0.5, 1.2))
        
    def testFullConnection(self):
        inlayer_input = scipy.zeros((1, 2), dtype='float64')
        inlayer_output = scipy.zeros((1, 2), dtype='float64')
        inlayer_inerror = scipy.zeros((1, 2), dtype='float64')
        inlayer_outerror = scipy.zeros((1, 2), dtype='float64')
        outlayer_input = scipy.zeros((1, 3), dtype='float64')
        outlayer_output = scipy.zeros((1, 3), dtype='float64')
        outlayer_inerror = scipy.zeros((1, 3), dtype='float64')
        outlayer_outerror = scipy.zeros((1, 3), dtype='float64')

        inlayer = cppbridge.SimpleLayer(
            'LinearLayer', 2, 
            inlayer_input, inlayer_output, 
            inlayer_inerror, inlayer_outerror)
            
        outlayer = cppbridge.SimpleLayer(
            'LinearLayer', 3,
            outlayer_input, outlayer_output, 
            outlayer_inerror, outlayer_outerror)
            
        parameters = scipy.array((0, 1, 2, 3, 4, 5), dtype='float64')
        derivatives = scipy.zeros(6, dtype='float64')
        
        con = cppbridge.FullConnection(
            inlayer.address, outlayer.address, 
            parameters, derivatives)
            
        inlayer_input[0, :] = 2., 3.
            
        inlayer.forward()
        self.assertArrayNear(inlayer_output[0], (2, 3))
        
        con.forward()
        self.assertArrayNear(outlayer_input[0], (3, 13, 23))

        outlayer.forward()
        self.assertArrayNear(outlayer_output[0], (3, 13, 23))

        outlayer_outerror[0, :] = 0.5, 1.2, 3.4
        
        outlayer.backward()
        con.backward()
        inlayer.backward()
        
        self.assertArrayNear(inlayer_inerror[0], (16, 21.1))
        self.assertArrayNear(derivatives, (1, 1.5, 2.4, 3.6, 6.8, 10.2))
        
        
class TestNetwork(TestCase):
    
    def testTwoLayerNetwork(self):
        network_input = scipy.zeros((1, 2), dtype='float64')
        network_inerror = scipy.zeros((1, 2), dtype='float64')
        network_output = scipy.zeros((1, 3), dtype='float64')
        network_outerror = scipy.zeros((1, 3), dtype='float64')
        
        inlayer_input = scipy.zeros((1, 2), dtype='float64')
        inlayer_output = scipy.zeros((1, 2), dtype='float64')
        inlayer_inerror = scipy.zeros((1, 2), dtype='float64')
        inlayer_outerror = scipy.zeros((1, 2), dtype='float64')
        
        outlayer_input = scipy.zeros((1, 3), dtype='float64')
        outlayer_output = scipy.zeros((1, 3), dtype='float64')
        outlayer_inerror = scipy.zeros((1, 3), dtype='float64')
        outlayer_outerror = scipy.zeros((1, 3), dtype='float64')

        inlayer = cppbridge.SimpleLayer(
            'LinearLayer', 2, 
            inlayer_input, inlayer_output, 
            inlayer_inerror, inlayer_outerror)
            
        outlayer = cppbridge.SimpleLayer(
            'LinearLayer', 3,
            outlayer_input, outlayer_output, 
            outlayer_inerror, outlayer_outerror)

        parameters = scipy.array((0, 1, 2, 3, 4, 5), dtype='float64')
        derivatives = scipy.zeros(6, dtype='float64')
        
        con = cppbridge.FullConnection(
            inlayer.address, outlayer.address, 
            parameters, derivatives)
            
        net = cppbridge.Network(network_input, network_output, 
                                network_inerror, network_outerror)
        net.add_module(inlayer.address, inpt=True)
        net.add_module(outlayer.address, outpt=True)    
        net.add_connection(con.address)
        
        net.activate((2, 3))
        
        self.assertArrayNear(network_input[0], (2, 3))
        self.assertArrayNear(inlayer_input[0], (2, 3))
        self.assertArrayNear(inlayer_output[0], (2, 3))
        self.assertArrayNear(outlayer_input[0], (3, 13, 23))
        self.assertArrayNear(outlayer_output[0], (3, 13, 23))
        self.assertArrayNear(network_output[0], (3, 13, 23))
        
        net.back_activate((0.5, 1.2, 3.4))
        
        self.assertArrayNear(network_outerror[0], (0.5, 1.2, 3.4))
        self.assertArrayNear(inlayer_inerror[0], (16, 21.1))
        self.assertArrayNear(network_inerror[0], (16, 21.1))
        self.assertArrayNear(derivatives, (1, 1.5, 2.4, 3.6, 6.8, 10.2))


if __name__ == "__main__":
    unittest.main()  