#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

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
        # See if there already is a proxy:
        try: 
            proxy = self.map[network]
            proxy.init_buffer('input', network.inputbuffer)
            proxy.init_buffer('output', network.outputbuffer)
            proxy.init_buffer('inerror', network.inputerror)
            proxy.init_buffer('outerror', network.outputerror)
        except KeyError:
            proxy = cppbridge.Network(
                        network.inputbuffer, network.outputbuffer, 
                        network.inputerror, network.outputerror)
        return proxy

    def _simple_layer_handler(self, layer):
        try:
            proxy = self.map[layer]
            proxy.init_buffer('input', layer.inputbuffer)
            proxy.init_buffer('output', layer.outputbuffer)
            proxy.init_buffer('inerror', layer.inputerror)
            proxy.init_buffer('outerror', layer.outputerror)
        except KeyError:
            s = str(type(layer))
            s = s[s.rfind('.') + 1:s.rfind("'")]
            proxy = cppbridge.SimpleLayer(
                        s, layer.dim, 
                        layer.inputbuffer, layer.outputbuffer, 
                        layer.inputerror, layer.outputerror)
        return proxy
        
    def _bias_handler(self, bias):
        return cppbridge.Bias()
        
    def _lstm_handler(self, layer):
        # See if there already is a proxy:
        try: 
            proxy = self.map[layer]
            proxy.init_buffer('input', layer.inputbuffer)
            proxy.init_buffer('output', layer.outputbuffer)
            proxy.init_buffer('inerror', layer.inputerror)
            proxy.init_buffer('outerror', layer.outputerror)
            proxy.init_buffer('state', layer.state)
            proxy.init_buffer('state_error', layer.stateError)
        except KeyError:
            proxy = cppbridge.LstmLayer(
                        layer.dim, 
                        layer.inputbuffer, layer.outputbuffer, layer.state,
                        layer.inputerror, layer.outputerror, layer.stateError)
        return proxy

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
            
    def _linear_connection_handler(self, con):
        try:
            incoming = self.map[con.inmod]
            outgoing = self.map[con.outmod]
        except KeyError, e:
            raise ValueError("Unknown module: %s" % e)
        return cppbridge.LinearConnection(
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
            LinearConnection: self._linear_connection_handler,
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

    @property
    def offset(self):
        return self.proxies[self].timestep()

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # Mapping the components of the network to their proxies.
        self.proxies = PybrainAracMapper()

    def _growBuffers(self):
        super(_Network, self)._growBuffers()
        self._rebuild()
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self._rebuild()

    def _rebuild(self):
        self.buildCStructure()
        
    def reset(self):
        net_proxy = self.proxies.handle(self)
        net_proxy.clear()

    def buildCStructure(self):
        """Build up a C++-network."""
        # We first add all the modules, since we have to know about them before
        # we can add connections.
        net_proxy = self.proxies.handle(self)
        
        for module in self.modules:
            add = not module in self.proxies
            mod_proxy = self.proxies.handle(module)
            if add:
                net_proxy.add_module(mod_proxy,
                                     inpt=(module in self.inmodules),
                                     outpt=(module in self.outmodules))
        for connectionlist in self.connections.values():
            for connection in connectionlist:
                add = not connection in self.proxies
                con_proxy = self.proxies.handle(connection)
                if add:
                    net_proxy.add_connection(con_proxy)
        
    def activate(self, inputbuffer):
        self.proxies[self].activate(inputbuffer)
        return self.outputbuffer[self.offset - 1]
        
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

    def activate(self, inputbuffer):
        while True:
            # Grow buffers until they have the correct size.
            if self.offset < self.outputbuffer.shape[0]:
                break
            # TODO: _growBuffers() is called more than once.
            self._growBuffers()
        result = _Network.activate(self, inputbuffer)
        return result

    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result

    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        _Network.buildCStructure(self)
        net_proxy = self.proxies[self]
        net_proxy.set_mode('Sequential')
        for connection in self.recurrentConns:
            add = not connection in self.proxies
            con_proxy = self.proxies.handle(connection)
            con_proxy.set_recurrent(1)
            if add:
                net_proxy.add_connection(con_proxy)
        for component in self.proxies.map.values():
            component.set_mode('Sequential')