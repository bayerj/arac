#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""This module provides a bridge from Python to arac.


Most classes in this module are proxies of the corresponding arac 
classes. They create the objects on initialization via scipy.weave and destroy
them when the python proxies are collected.

Some methods of the arac classes are proxied, but not everyone. For a complete 
list, see the source code.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy.weave
import scipy.weave.converters


def arac_call(code, namespace=None):
    if namespace is None:
        namespace = {}
    support_code = """
    #include "/Users/bayerj/devel/arac0.3/src/cpp/arac.h"
    
    using namespace arac::structure::networks;
    using namespace arac::structure::modules;
    using namespace arac::structure::connections;
    """
    libraries = ['arac']
    type_converters = scipy.weave.converters.blitz
    result = scipy.weave.inline(code, namespace.keys(), namespace,
                       include_dirs=['/Users/bayerj/devel/arac0.3/src/cpp'],
                       library_dirs=['/Users/bayerj/devel/arac0.3'],
                       support_code=support_code,
                       libraries=libraries,
                       type_converters=type_converters
    )
    return result


class Proxy(object):
    """Subclass for all representatives of arac objects."""
    
    typ = None
    address = 0
    
    def __del__(self):
        if self.address:
            code = "delete (%s*) address;" % self.typ
            arac_call(code, {'address': self.address})
        
    def pcall(self, code, namespace):
        """Call the passed code; a pointer to the proxied object is already
        available as 'p' in the code."""
        code = "%(typ)s* p = (%(typ)s*) address; \n" + code
        namespace['address'] = self.address
        return arac_call(code, namespace)


class Component(Proxy):
    
    def __init__(self):
        code = """
        %(typ)s* p = new %(typ)s();
        return_val = (int) p;
        """ % {'typ': self.typ}
        self.address = arac_call(code)
    
    def forward(self):
        code = "p->forward();"
        self.pcall(code)

    def backward(self):
        code = "p->backward();"
        self.pcall(code)
    
    
class Parametrized(Proxy):

    def set_parameters(self, arr):
        parameters = arr.ctypes.data
        code = "p->set_parameters((double*) parameters);"
        self.pcall(code, {'parameters': parameters})
    

class Module(Component):

    def __init__(self, inpt, outpt, inerror=None, outerror=None):
        # TODO: implement! ;)
        pass

    def append_to_buffer(self, buffername, pointer):
        """Append a double pointer to a specified buffer."""
        code = "p->%s().append((double*) pointer)" % buffername
        self.pcall(code, {'pointer': pointer})


class Connection(Component):
    
    typ = 'Connection'
    
    def __init__(self, incoming, outgoing, 
                 incomingstart=None, incomingstop=None, 
                 outgoingstart=None, outgoingstop=None):
        slices = incomingstart, incomingstop, outgoingstart, outgoingstop
        if any(i is None for i in slices):
            if not all(i is None for i in slices):
                raise ValueError("Either specify all or no slice.") 
            # Code for sliced connections.
            code =  """
                    %(typ)s* p = new %(typ)s(incoming, outgoing);
                    return_val = (int) p;
                    """ % {'typ': self.typ}
            self.address = arac_call(code, {'incoming': incoming, 
                                            'outgoing': outgoing})
        else:
            # Code for not sliced connections.
            code =  """
            %(typ)s* p = new %(typ)s(incoming, outgoing, 
                                     incomingstart, incomingstop, 
                                     outgoingstart, outgoingstop);
            return_val = (int) p;
            """ % {'typ': self.typ}
            self.address = arac_call(code, {'incoming': incoming,
                                            'outgoing': outgoing,
                                            'incomingstart': incomingstart,
                                            'incomingstop': incomingstop,
                                            'outgoingstart': outgoingstart,
                                            'outgoingstop': outgoingstop})
            
    
class IdentityConnection(Component):
    
    typ = 'IdentityConnection'
    
    def __init__(self, incoming, outgoing, 
                 incomingstart=None, incomingstop=None, 
                 outgoingstart=None, outgoingstop=None):
        super(IdentityConnection, self).__init__(
            incoming, outgoing, 
            incomingstart, incomingstop, 
            outgoingstart, outgoingstop)
    

class FullConnection(Connection):
    
    typ = 'FullConnection'
    
    def __init__(self, incoming, outgoing, parameters, derivatives,
                 incomingstart=None, incomingstop=None, 
                 outgoingstart=None, outgoingstop=None):
        super(FullConnection, self).__init__(incoming, outgoing, 
            incomingstart, incomingstop, 
            outgoingstart, outgoingstop)
        code = """
        p->set_parameters(parameters_p);
        p->set_derivatives(derivatives_p);
        """
        self.pcall(code, {'parameters_p': parameters.ctypes.data,
                          'derivatives_p': derivatives_p.ctypes.data})
        

class BaseNetwork(Module):
    
    def activate(self, arr):
        return self.pcall('p->activate(input_p);', {'input_p': arr.ctypes.data})
        
    def back_activate(self, arr):
        return self.pcall('p->back_activate(error_p);', 
                          {'error_p': arr.ctypes.data})
                          
    def sort(self):
        return self.pcall('p->sort();')

    
class Network(BaseNetwork):
    
    def add_module(self, module_p, input=False, output=False):
        inputoutput = {
            (False, False): 0,
            (True, False): 1,
            (False, True): 2,
            (True, True): 3
        }
        self.pcall('p->add_module(module_p, inputoutput);',
                   {'module_p': module_p,
                    'inputoutput': inputoutput})
                    
    def add_connection(self, con_p):
        self.pcall('p->add_connection(con_p)', {'con_p': con_p})
        
    def clear(self):
        self.pcall('p->clear();')
        
    def sort(self):
        self.pcall('p->sort();')
        
        
class Mdrnn(BaseNetwork):
    
    def __init__(self, layerklass, timedim, hiddensize):
        pass
        

class SimpleLayer(Module):
    
    def __init__(self, klass, size, inpt, outpt, inerror=None, outerror=None):
        if not klass.isalnum():
            raise ValueError("Wrong layer identifier.")
        super(SimpleLayer, self).__init__(inpt, outpt, inerror, outerror)
        self.size = size
        self.klass = klass
        self.inpt = inpt
        self.outpt = outpt
        self.inerror = inerror
        self.outerror = outerror
        self.init_layer()

    def init_layer(self):
        code = """
        %(klass)s* layer_p = new %(klass)s(size);
        return_val = (int) layer_p;
        """ % {'klass': self.klass}
        self.address = arac_call(code, {'size': self.size})