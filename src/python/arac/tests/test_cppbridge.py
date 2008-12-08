#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittests for the arac.cppbridge module.

    >>> 2
    2

"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import arac.cppbridge as cppbridge
import scipy
import unittest

from arac.tests.common import array_equal

class AracCall(unittest.TestCase):
    
    def testCall(self):
        result = cppbridge.arac_call("""return_val = 1;""")
        self.assertEqual(result, 1)


class TestSimpleLayer(unittest.TestCase):
    
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
        inpt = scipy.empty(2)
        outpt = scipy.empty(2)
        inerror = scipy.empty(2)
        outerror = scipy.empty(2)
        layer = cppbridge.SimpleLayer(klass, 2, inpt, outpt, inerror, outerror)
        inpt[:] = conf['inpt']
        layer.forward()
        self.assert_(array_equal(outpt, conf['outpt']))
        layer.backward()
        self.assert_(array_equal(inerror, conf['inerror']))
    
    def testLinearLayer(self):
        self.run_configuration('LinearLayer')

    def testSigmoidLayer(self):
        self.run_configuration('SigmoidLayer')

    def testTanhLayer(self):
        self.run_configuration('TanhLayer')

    def testSoftmaxLayer(self):
        self.run_configuration('SoftmaxLayer')


if __name__ == "__main__":
    unittest.main()  