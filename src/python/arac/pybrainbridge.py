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
    BiasUnit,
    LinearLayer, 
    LSTMLayer,
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

from pybrain.structure.networks.feedforward import \
    FeedForwardNetworkComponent, FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetworkComponent, \
    RecurrentNetwork
    
import arac.cppbridge as cppbridge


class PybrainAracMapper(cppbridge.ProxyContainer):
    """Class that holds pybrain objects mapped to arac objects and provides 
    handlers to create new ones from pybrain objects."""

    def _network_handler(self, network):
        return cppbridge.Network(
            network.inputbuffer, network.outputbuffer, 
            network.inputerror, network.outputerror)

    def _simple_layer_handler(self, layer):
        s = str(type(layer))
        s = s[s.rfind('.') + 1:s.rfind("'")]
        return cppbridge.SimpleLayer(
            s, layer.dim, 
            layer.inputbuffer, layer.outputbuffer, 
            layer.inputerror, layer.outputerror)
        
    def _bias_handler(self, bias):
        return cppbridge.Bias()
        
    def _lstm_handler(self, layer):
        return cppbridge.LstmLayer(layer.dim, 
                                   layer.inputbuffer, layer.outputbuffer, layer.state,
                                   layer.inputerror, layer.outputerror, layer.stateError)
        
    def _full_connection_handler(self, con):
        try:
            incoming = self.map[con.inmod]
            outgoing = self.map[con.outmod]
        except KeyError, e:
            raise ValueError("Unknown module: %s" % e)
        return cppbridge.FullConnection(
            incoming, outgoing, 
            con.params, con.derivs,
            con.inSliceFrom, con.inSliceTo,
            con.outSliceFrom, con.outSliceTo)
            
    def _identity_connection_handler(self, con):
        incoming = self.map[con.inmod]
        outgoing = self.map[con.outmod]
        return cppbridge.IdentityConnection(
            incoming, outgoing, 
            con.inSliceFrom, con.inSliceTo,
            con.outSliceFrom, con.outSliceTo)
        
        
    def handle(self, obj):
        handlers = {
            BiasUnit: self._bias_handler,
            LinearLayer: self._simple_layer_handler, 
            LSTMLayer: self._lstm_handler,
            SigmoidLayer: self._simple_layer_handler, 
            SoftmaxLayer: self._simple_layer_handler,
            TanhLayer: self._simple_layer_handler,
            IdentityConnection: self._identity_connection_handler, 
            FullConnection: self._full_connection_handler,
            Network: self._network_handler,
            RecurrentNetwork: self._network_handler,
            FeedForwardNetwork: self._network_handler,
            _FeedForwardNetwork: self._network_handler,
            _RecurrentNetwork: self._network_handler,
        }
        self.map[obj] = handlers[type(obj)](obj)
        return self.map[obj]
    

class _Network(Network):
    """Adapter for the pybrain Network class.

    Currently, the only way to process input to the network is the .activate()
    method. The only way to propagate the error back is .backActivate().
    """

    offset = 0

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # Mapping the components of the network to their proxies.

    def reset(self):
        self.proxies[self].clear()
        
    def _growBuffers(self):
        super(_Network, self)._growBuffers()
        self._rebuild()
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self._rebuild()

    def _rebuild(self):
        self.buildCStructure()

    def buildCStructure(self):
        """Build up a C++-network."""
        # We first add all the modules, since we have to know about them before
        # we can add connections.
        self.proxies = PybrainAracMapper()
        net_proxy = self.proxies.handle(self)
        
        for module in self.modules:
            mod_proxy = self.proxies.handle(module)
            net_proxy.add_module(mod_proxy,
                                 inpt=(module in self.inmodules),
                                 outpt=(module in self.outmodules))
        for connectionlist in self.connections.values():
            for connection in connectionlist:
                con_proxy = self.proxies.handle(connection)
                net_proxy.add_connection(con_proxy)
        
    def activate(self, inputbuffer):
        self.proxies[self].activate(inputbuffer)
        return self.outputbuffer[self.offset]
        
    def backActivate(self, outerr):
        self.proxies[self].back_activate(outerr)
        return self.inputerror[self.offset]

        
class _FeedForwardNetwork(FeedForwardNetworkComponent, _Network):
    """Pybrain adapter for an arac FeedForwardNetwork."""

    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        FeedForwardNetworkComponent.__init__(self, *args, **kwargs)
        
    def activate(self, inputbuffer):
        result = _Network.activate(self, inputbuffer)
        return result
        
    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result

        
class _RecurrentNetwork(RecurrentNetworkComponent, _Network):
    """Pybrain adapter for an arac RecurrentNetwork."""

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
        net_proxy = self.proxies[self]
        for connection in self.recurrentConns:
            con_proxy = self.proxies.handle(connection)
            con_proxy.set_recurrent(1)
            net_proxy.add_connection(con_proxy)