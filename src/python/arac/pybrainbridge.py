#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""This module provides a bridge from arac to PyBrain.

Arac features ways to compose networks similar to PyBrain. Since arac is C++ 
which is not easily usable from python, inline weave is used to construct an arac
network in parallel.

To ease this, this module features the classes _FeedForwardNetwork and
_RecurrentNetwork, which mimic the behaviour of the PyBrain API. Whenever you
want to use arac's speed benefits, use these classes instead of its pybrain
counterparts.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from pybrain.structure import (
    LinearLayer, 
    SigmoidLayer, 
    TanhLayer,
    SoftmaxLayer,
    PartialSoftmaxLayer,
    IdentityConnection, 
    FullConnection,
    Network,
    RecurrentNetwork,
    FeedForwardNetwork
)
    
import arac.cppbridge as cppbridge


class PybrainAracMapper(object):
    """Class that holds pybrain objects mapped to arac objects and provides 
    handlers to create new ones from pybrain objects."""

    def __init__(self):
        self.map = {}

    def _network_handler(self, network):
        # TODO: set buffers!
        return cppbridge.Network()

    def _simple_layer_handler(self, layer):
        return cppbridge.SimpleLayer(
            type(layer), layer.dim, 
            layer.inputbuffer, layer.outputbuffer, 
            layer.inputerror, layer.outputerror)
        
    def _full_connection_handler(self, con):
        try:
            incoming = self.map[con.inmod]
            outgoing = self.map]con.outmod]
        except KeyError, e:
            raise ValueError("Unknown module: %s" % e)
        return cppbridge.FullConnection(
            incoming, outgoing, 
            con.parameters.ctypes.data, con.derivatives.ctypes.data,
            con.inSliceFrom, con.inSliceTo,
            con.outSliceFrom, con.outSliceTo)
            
    def _identity_connection_handler(self, con):
        incoming = self.map[con.inmod]
        outgoing = self.map]con.outmod]
        return cppbridge.IdentityConnection(
            incoming, outgoing, 
            con.inSliceFrom, con.inSliceTo,
            con.outSliceFrom, con.outSliceTo)
        
    def handle(self, obj):
        handlers = {
            LinearLayer: self._simple_layer_handler, 
            SigmoidLayer: self._simple_layer_handler, 
            TanhLayer: self._simple_layer_handler,
            SoftmaxLayer: self._simple_layer_handler,
            PartialSoftmaxLayer: self._simple_layer_handler,
            IdentityConnection: self._identity_connection_handler, 
            FullConnection: self._full_connection_handler
            Network: self._network_handler,
            RecurrentNetwork: self._network_handler,
            FeedForwardNetwork: self._network_handler,

        }
        self.map[obj] = handlers(type(obj))(obj)
        return self.map[obj]
    

class _Network(Network):
    """Adapter for the pybrain Network class.

    Currently, the only way to process input to the network is the .activate()
    method. The only way to propagate the error back is .backActivate().
    """

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # Mapping the components of the network to their proxies.
        self.proxies = PybrainAracMapper()

    def reset(self):
        self.proxies[self].clear()
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self.proxies[self].sort()

    def _rebuild(self):
        self.buildCStructure()

    def buildCStructure(self):
        """Build up a C++-network."""
        # First make a struct for every module.
        root = None
        proxies = {}
        for module in self.modules:
            # Get handler.
            self.proxies.handle(modules)

        for connection in connections:
            self.proxies.handle(connection)
            # TODO: take care of recurrency here.
        
    def activate(self, inputbuffer):
        # The outputbuffer of the first module in the list is which we decide
        # upon wether we have to grow the buffers.
        # This relies upon the fact, that all buffers of the network have the
        # same size (in terms of timesteps).
        while True:
            # Grow buffers until they have the correct size, so possibly call
            # _growBuffers() more than once.
            # TODO: get correct offset here.
            if self.offset < self.outputbuffer.shape[0]:
                break
            self._growBuffers()
            # FIXME: only rebuild once in the end.
            self._rebuild()

        self.proxies[self].activate(inputbuffer)
        # FIXME: return the right thing
        
    def backActivate(self, outerr):
        # Function libarac.calc_derivs decrements the offset, so we have to
        # compensate for that here first.
        self.proxies[self].back_activate(outerr)
        # FIXME: return the right thing
        
        
class _FeedForwardNetwork(FeedForwardNetworkComponent, _Network):
    """Counterpart to pybrain's FeedForwardNetwork."""

    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        FeedForwardNetworkComponent.__init__(self, *args, **kwargs)
        
    # Arac always increments/decrements the offset after 
    # activation/backactivation. In the case of FFNs, we have to compensate for
    # this.
    
    def activate(self, inputbuffer):
        self.reset()
        result = _Network.activate(self, inputbuffer)
        return result
        
    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result
        
class _RecurrentNetwork(RecurrentNetworkComponent, _Network):
    """Counterpart to pybrain's RecurrentNetwork."""

    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        RecurrentNetworkComponent.__init__(self, *args, **kwargs)
        # TODO: make the network a sequential one.

    def activate(self, inputbuffer):
        result = _Network.activate(self, inputbuffer)
        return result

    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result

    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        _Network.buildCStructure(self)
        for connection in self.recurrentConns:
            struct_connection = self.buildCStructForConnection(connection)
            struct_connection.recurrent = 1
            key = '%s-%i' % (connection.name, id(connection))
            self.cconnections[key] = struct_connection