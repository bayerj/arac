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
                       type_converters=type_converters,
                       extra_compile_args=['-g']
    )
    return result


class ProxyContainer(object):
    """Class that handles proxies and deletes the held objects of them when the
    container is deleted."""
    
    def __init__(self):
        self.clear()
        
    def __del__(self):
        for proxy in self.map.items():
            proxy.free()
        
    def __getitem__(self, key):
        return self.map[key]
        
    def clear(self):
        """Free the current map and all the held structures."""
        self.map = {}



class Proxy(object):
    """Subclass for all representatives of arac objects."""
    
    typ = None
    address = 0
    
    def free(self):
        if self.address:
            code = "delete (%s*) address;" % self.typ
            arac_call(code, {'address': self.address})
        
    def pcall(self, code, namespace=None):
        """Call the passed code; a pointer to the proxied object is already
        available as 'p' in the code."""
        if namespace is None:
            namespace = {}
        code = "%(typ)s* p = (%(typ)s*) address; \n" % {'typ': self.typ} + code 
        # print code
        # print namespace
        # print
        namespace['address'] = self.address
        return arac_call(code, namespace)


class Component(Proxy):
    
    def __init__(self):
        if self.typ is None:
            raise ValueError("Attribute .typ has not been set.")
        self._init_object()
            
    def _init_object(self):
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
        super(Module, self).__init__()
        self.inpt = inpt
        self.outpt = outpt
        self.inerror = inerror
        self.outerror = outerror
        self.init_buffers()
        
    def init_buffers(self):
        self.init_buffer('input', self.inpt)
        self.init_buffer('output', self.outpt)
        self.init_buffer('inerror', self.inerror)
        self.init_buffer('outerror', self.outerror)
        
    def init_buffer(self, buffername, arr):
        self.pcall("p->%s().free_memory();" % buffername)
        for row in arr:
            self.append_to_buffer(buffername, row.ctypes.data)

    def append_to_buffer(self, buffername, pointer):
        """Append a double pointer to a specified buffer."""
        code = "p->%s().append((double*) pointer);" % buffername
        self.pcall(code, {'pointer': pointer})
        
        
class SimpleLayer(Module):
    
    def __init__(self, typ, size, inpt, outpt, inerror=None, outerror=None):
        if not typ.isalnum():
            raise ValueError("Wrong layer identifier.")
        self.typ = typ
        self.size = size
        super(SimpleLayer, self).__init__(inpt, outpt, inerror, outerror)

    def _init_object(self):
        code = """
        %(typ)s* layer_p = new %(typ)s(size);
        return_val = (int) layer_p;
        """ % {'typ': self.typ}
        self.address = arac_call(code, {'size': self.size})
        
        
class Bias(Module):
    
    typ = 'Bias'
    
    def __init__(self):
        Component.__init__(self)


class LstmLayer(SimpleLayer):
    
    typ = 'LstmLayer'
    
    def __init__(self, size, 
                 inpt, outpt, state,
                 inerror=None, outerror=None, state_error=None):
        self.state = state
        self.state_error = state_error
        super(LstmLayer, self).__init__(
            self.typ, size, inpt, outpt, inerror, outerror)
        
    def init_buffers(self):
        self.init_buffer('state', self.state)
        self.init_buffer('state_error', self.state_error)
        super(LstmLayer, self).init_buffers()


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
                    %(typ)s* p = new %(typ)s((%(intype)s*) incoming, 
                                             (%(outtype)s*) outgoing);
                    return_val = (int) p;
                    """ % {'typ': self.typ, 
                           'intype': incoming.typ, 
                           'outtype': outgoing.typ}
            self.address = arac_call(code, {'incoming': incoming.address, 
                                            'outgoing': outgoing.address})
        else:
            # Code for not sliced connections.
            code =  """
            %(typ)s* p = new %(typ)s((%(intype)s*) incoming, 
                                     (%(outtype)s*) outgoing, 
                                     incomingstart, incomingstop, 
                                     outgoingstart, outgoingstop);
            return_val = (int) p;
            """ % {'typ': self.typ, 
                   'intype': incoming.typ, 
                   'outtype': outgoing.typ}
            self.address = arac_call(code, {'incoming': incoming.address,
                                            'outgoing': outgoing.address,
                                            'incomingstart': incomingstart,
                                            'incomingstop': incomingstop,
                                            'outgoingstart': outgoingstart,
                                            'outgoingstop': outgoingstop})
    def set_recurrent(self, value):
        self.pcall('p->set_recurrent(value);', {'value': value})
            
    
class IdentityConnection(Connection):
    
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
        p->set_parameters((double*) parameters_p);
        p->set_derivatives((double*) derivatives_p);
        """
        self.pcall(code, {'parameters_p': parameters.ctypes.data,
                          'derivatives_p': derivatives.ctypes.data})
        

class BaseNetwork(Module):
    
    def activate(self, arr):
        if type(arr) != scipy.ndarray:
            arr = scipy.array(arr, dtype='float64')
        self.pcall('p->activate((double*) input_p);', 
                   {'input_p': arr.ctypes.data})
        
    def back_activate(self, arr):
        if type(arr) != scipy.ndarray:
            arr = scipy.array(arr)
        self.pcall('p->back_activate((double*) error_p);', 
                   {'error_p': arr.ctypes.data})

    
class Network(BaseNetwork):
    
    typ = 'Network'
    
    def add_module(self, module, inpt=False, outpt=False):
        if not isinstance(module, Module):
            raise ValueError(
                "module has to be Module Proxy, not %s" % type(module))
        inputoutput = {
            (False, False): 0,
            (True, False): 1,
            (False, True): 2,
            (True, True): 3
        }
        code = """
        p->add_module((%s*) module_p, (Network::ModuleType) inputoutput);
        """  % module.typ
        self.pcall(code, 
                   {'module_p': module.address,
                    'inputoutput': inputoutput[(inpt, outpt)]})
                    
    def add_connection(self, con):
        if not isinstance(con, Connection):
            raise ValueError("con has to be Connection Proxy.")
        self.pcall('p->add_connection((%s*) con_p);' % con.typ, 
                   {'con_p': con.address})
        
    def clear(self):
        self.pcall('p->clear();')
        
        
class Mdrnn(BaseNetwork):
    
    def __init__(self, layerklass, timedim, hiddensize):
        pass
        

