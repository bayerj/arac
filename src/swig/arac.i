%module arac
%{
#define SWIG_FILE_WITH_INIT
    
#include "../cpp/arac.h"
#include <numpy/arrayobject.h>

using namespace arac::structure;
using namespace arac::structure::connections;
using namespace arac::structure::modules;
using namespace arac::structure::networks;
%}

%include "numpy.i"
%init %{
    import_array();
%}


class Component 
{
    public: 
        enum Mode 
        {
            Simple = 0,
            ErrorAgnostic = 1,
            Sequential = 2,
            
            SequentialErrorAgnostic = 3,
        };
        virtual Component();
        virtual ~Component();
        virtual void forward();
        virtual void backward();
        virtual void dry_forward();
        virtual void dry_backward();
        virtual void set_mode(Mode mode);
        virtual void clear();
        Mode get_mode();
        bool sequential();
        int timestep();
        int sequencelength();
        bool error_agnostic();
    protected:
        virtual void _forward() = 0;
        virtual void _backward() = 0;

};


class Module : public Component
{
    public:
        Module();
        Module(int insize, int outsize);
        virtual ~Module();
        virtual void forward();
        void add_to_input(double* addend_p);
        void add_to_outerror(double* addend_p);
        virtual void clear();
        arac::common::Buffer& input();
        arac::common::Buffer& output();
        arac::common::Buffer& inerror();
        arac::common::Buffer& outerror();
        int insize();
        int outsize();
        bool last_timestep();
};


class Parametrized 
{
    public: 
        Parametrized();
        Parametrized(int size);
        virtual ~Parametrized();
        double* get_parameters() const;
        void set_parameters(double* parameters_p);
        double* get_derivatives() const;
        void set_derivatives(double* derivatives_p);
        void clear_derivatives();
};


class Connection : public Component
{
    public: 
        Connection(Module* incoming, Module* outgoing,
                   int incomingstart, int incomingstop, 
                   int outgoingstart, int outgoingstop);
        Connection(Module* incoming, Module* outgoing);
        virtual ~Connection();
        
        void set_incomingstart(int n);
        void set_incomingstop(int n);
        void set_outgoingstart(int n);
        void set_outgoingstop(int n);
        
        int get_incomingstart();
        int get_incomingstop();
        int get_outgoingstart();
        int get_outgoingstop();
        
        void set_recurrent(int recurrent);
        int get_recurrent();
        
        Module* incoming();
        Module* outgoing();
};


%feature("notabstract") IdentityConnection;
class IdentityConnection : public Connection 
{
    public:
        IdentityConnection(Module* incoming_p, Module* outgoing_p);
        IdentityConnection(Module* incoming_p, Module* outgoing_p,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop);
        virtual ~IdentityConnection();
};


%feature("notabstract") LinearLayer;
class LinearLayer : public Module
{
    public:
        LinearLayer(int size);
        virtual ~LinearLayer();
};


%feature("notabstract") SigmoidLayer;
class SigmoidLayer : public Module
{
    public:
        SigmoidLayer(int size);
        virtual ~LinearLayer();
};


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* input_p, int inlength), 
                                           (double* output_p, int outlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* outerror_p, int outlength), 
                                           (double* inerror_p, int inlength)};

class BaseNetwork : public Module
{
    
    public:
        BaseNetwork();
        virtual ~BaseNetwork();
    
        virtual void activate(double* input_p, double* output_p);
        virtual void back_activate(double* outerror_p, double* inerror_p);
        virtual void forward();
        
    protected:
        virtual void sort() = 0;
};


%feature("notabstract") Network;
class Network : public BaseNetwork
{
    public: 
        
        enum ModuleType {
            Simple = 0,
            InputModule = 1,
            OutputModule = 2,
            InputOutputModule = 3
        };
        Network();
        virtual ~Network();
        virtual void clear();
        void add_module(Module* module_p, ModuleType type=Simple);
        void add_connection(Connection* con_p);
};        
        
        
%extend Network
{
    virtual void activate(double* input_p, int inlength, 
                          double* output_p, int outlength)
    {
        if (inlength != outlength) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         inlength, outlength);
            return;
        }
        $self->activate(input_p, output_p);
    }

    virtual void back_activate(double* outerror_p, int outlength, 
                               double* inerror_p, int inlength)
    {
        if (inlength != outlength) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         inlength, outlength);
            return;
        }
        $self->back_activate(outerror_p, inerror_p);
    }
};