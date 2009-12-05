%module cppbridge
%{
#define SWIG_FILE_WITH_INIT
    
#include <iostream>
#include <vector>
#include <cassert>

#include <numpy/arrayobject.h>

#include "../cpp/arac.h"


using namespace arac::common;
using namespace arac::datasets;
using namespace arac::optimization;
using namespace arac::structure;
using namespace arac::structure::connections;
using namespace arac::structure::modules;
using namespace arac::structure::networks;
using namespace arac::structure::networks::mdrnns;
using namespace arac::utilities;


void init_buffer(Buffer& buffer, double* content_p, int length, int rowsize)
{
    buffer.free_memory();
    buffer.set_rowsize(rowsize);
    bool prev = buffer.contmemory();
    for(int i = 0; i < length; i++)
    {
        buffer.append(content_p + i * rowsize);
    }
}


PyObject* PyArray_1DFromDoublePointer(int dim, double* data_p)
{
    int* dims = new int[1];
    dims[0] = dim;
    // TODO: use python function that does not trigger a warning here.
    PyObject* res_p = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE, 
                                              (char*) data_p);
    delete[] dims;
    return res_p;
}


PyObject* PyArray_2DFromDoublePointer(int dim1, int dim2, double* data_p)
{
    int* dims = new int[2];
    dims[0] = dim1;
    dims[1] = dim2;
    // TODO: use python function that does not trigger a warning here.
    PyObject* res_p = PyArray_FromDimsAndData(2, dims, PyArray_DOUBLE, 
                                              (char*) data_p);
    delete[] dims;
    return res_p; 
}


%}


%include "typemaps.i"
%include "std_vector.i"
namespace std
{
    %template(VectorParametrized) std::vector<Parametrized*>;
    %template(VectorBaseNetwork) std::vector<BaseNetwork*>;
    %template(VectorInt) std::vector<int>;
    %template(VectorDouble) std::vector<double>;
}

%include "numpy.i"
%init %{
    import_array();
%}


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* input_p, int inlength), 
                                           (double* output_p, int outlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* outerror_p, int outlength), 
                                           (double* inerror_p, int inlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* parameters_p, int parameter_size), 
                                           (double* derivatives_p, int derivative_size)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* outerror_p, int outlength), 
                                           (double* inerror_p, int inlength)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* content_p, int length, int rowsize)};


%nodefaultctor Buffer;
class Buffer
{
    // Add the given pointer as a row.
    void append(double* row);
};


%extend Buffer 
{
    void append(double* row_p, int this_size)
    {
        if (this_size != $self->rowsize()) {
            PyErr_Format(PyExc_ValueError, "Row has wrong length: (%d,%d) given",
                         this_size, $self->rowsize());
            return;
        }
        $self->append(row_p);
    }
};


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


%extend Module 
{
    void init_input(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input(), content_p, length, rowsize);
    }

    void init_output(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output(), content_p, length, rowsize);
    }
    
    void init_inerror(double* content_p, int length, int rowsize)
    {
        init_buffer($self->inerror(), content_p, length, rowsize);
    }

    void init_outerror(double* content_p, int length, int rowsize)
    {
        init_buffer($self->outerror(), content_p, length, rowsize);
    }
}


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* parameters_p, int n_parameters)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* derivatives_p, int n_derivatives)};

class Parametrized 
{
    public: 
        Parametrized();
        Parametrized(int size);
        int size();
        virtual ~Parametrized();
        void clear_derivatives();
};


%extend Parametrized
{
    PyObject* get_parameters()
    {
        return PyArray_1DFromDoublePointer($self->size(), $self->get_parameters());
    }
    
    void set_parameters(double* parameters_p, int n_parameters)
    {
        if (n_parameters != $self->size())
        {
            PyErr_Format(PyExc_ValueError, "Arrays of length (%d) given",
                         n_parameters);
            return;
        }
        $self->set_parameters(parameters_p);
    }
    
    PyObject* get_derivatives()
    {
        return PyArray_1DFromDoublePointer($self->size(), $self->get_derivatives());
    }
    
    void set_derivatives(double* derivatives_p, int n_derivatives)
    {
        if (n_derivatives != $self->size())
        {
            PyErr_Format(PyExc_ValueError, "Arrays of length (%d) given",
                         n_derivatives);
            return;
        }
        $self->set_derivatives(derivatives_p);
    }
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


%feature("notabstract") Bias;
class Bias : public Module
{
    public:
        Bias();
        virtual ~Bias();
};


%feature("notabstract") GateLayer;
class GateLayer : public Module
{
    public:
        GateLayer(int size);
        virtual ~GateLayer();
};


%feature("notabstract") DoubleGateLayer;
class DoubleGateLayer : public Module
{
    public:
        DoubleGateLayer(int size);
        virtual ~DoubleGateLayer();
};


%feature("notabstract") MultiplicationLayer;
class MultiplicationLayer : public Module
{
    public:
        MultiplicationLayer(int size);
        virtual ~MultiplicationLayer();
};


%feature("notabstract") SwitchLayer;
class SwitchLayer : public Module
{
    public:
        SwitchLayer(int size);
        virtual ~SwitchLayer();
};


%feature("notabstract") LinearLayer;
class LinearLayer : public Module
{
    public:
        LinearLayer(int size);
        virtual ~LinearLayer();
};


%feature("notabstract") ErrorScalingLayer;
class ErrorScalingLayer : public Module
{
    public:
        ErrorScalingLayer(int size, std::vector<double> scale);
        virtual ~ErrorScalingLayer();
};


%feature("notabstract") LstmLayer;
class LstmLayer : public Module
{
    public:
        LstmLayer(int size);
        virtual ~LstmLayer();
};


%extend LstmLayer {

    void init_state(double* content_p, int length, int rowsize)
    {
        init_buffer($self->state(), content_p, length, rowsize);
    }

    void init_state_error(double* content_p, int length, int rowsize)
    {
        init_buffer($self->state_error(), content_p, length, rowsize);
    }
}


%feature("notabstract") MdlstmLayer;
class MdlstmLayer : public Module
{
    public:
        MdlstmLayer(int timedim, int size);
        virtual ~MdlstmLayer();
};


%extend MdlstmLayer {

    void init_input_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_squashed(), content_p, length, rowsize);
    }
    
    void init_input_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_gate_squashed(), content_p, length, rowsize);
    }
    
    void init_input_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_gate_unsquashed(), content_p, length, rowsize);
    }
    
    void init_output_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output_gate_squashed(), content_p, length, rowsize);
    }
    
    void init_output_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output_gate_unsquashed(), content_p, length, rowsize);
    }
        
    void init_forget_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->forget_gate_unsquashed(), content_p, length, rowsize);
    }
    
    void init_forget_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->forget_gate_squashed(), content_p, length, rowsize);
    }
}


%feature("notabstract") PartialSoftmaxLayer;
class PartialSoftmaxLayer : public Module
{
    public:
        PartialSoftmaxLayer(int size, int slicelength);
        virtual ~PartialSoftmaxLayer();
};


%feature("notabstract") SigmoidLayer;
class SigmoidLayer : public Module
{
    public:
        SigmoidLayer(int size);
        virtual ~SigmoidLayer();
};

%feature("notabstract") CosineLayer;
class CosineLayer : public Module
{
    public:
        CosineLayer(int size);
        virtual ~CosineLayer();
};

%feature("notabstract") SoftmaxLayer;
class SoftmaxLayer : public Module
{
    public:
        SoftmaxLayer(int size);
        virtual ~SoftmaxLayer();
};


%feature("notabstract") TanhLayer;
class TanhLayer : public Module
{
    public:
        TanhLayer(int size);
        virtual ~TanhLayer();
};


class BaseNetwork : public Module
{
    
    public:
        BaseNetwork();
        virtual ~BaseNetwork();
    
        virtual void activate(double* input_p, double* output_p);
        virtual void back_activate(double* outerror_p, double* inerror_p);
        virtual void forward();
        
        std::vector<Parametrized*>& parametrizeds();
        
        std::vector<BaseNetwork*>& networks();
        
        virtual void sort() = 0;
        virtual void randomize();
};


%extend BaseNetwork
{
    // TODO: do not make the API request a output array, but return a PyObject*
    // instead.
    
    virtual void activate(double* input_p, int inlength, 
                          double* output_p, int outlength)
    {
        // TODO: check bounds of in and output
        // if (inlength != $self->insize()) {
        //     PyErr_Format(PyExc_ValueError, 
        //                  "Input has wrong size: %d instead of %d",
        //                  inlength, $self->insize());
        //     return;
        // }
        // if (outlength != $self->outsize()) {
        //     PyErr_Format(PyExc_ValueError, 
        //                  "Output has wrong size: %d instead of %d",
        //                  outlength, $self->outsize());
        //     return;
        // }

        $self->activate(input_p, output_p);
    }

    virtual void back_activate(double* outerror_p, int outlength, 
                               double* inerror_p, int inlength)
    {
        if (inlength != $self->insize() or outlength != $self->outsize()) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         inlength, outlength);
            return;
        }
        $self->back_activate(outerror_p, inerror_p);
    }
};


%feature("notabstract") FullConnection;
class FullConnection : public Connection, public Parametrized
{
    public: 
        
        FullConnection(Module* incoming_p, Module* outgoing_p);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~FullConnection();
        
        %extend 
        {
            FullConnection(Module* incoming_p, Module* outgoing_p,
                           double* parameters_p, int parameter_size,
                           double* derivatives_p, int derivative_size,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop)
    
            {
                int required_size = \
                    (incomingstop - incomingstart) * (outgoingstop - outgoingstart);
                if (parameter_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Parameters have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                if (derivative_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Derivatives have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
        
                FullConnection* con = new FullConnection(incoming_p, outgoing_p, 
                                                         parameters_p, derivatives_p,
                                                         incomingstart, incomingstop,
                                                         outgoingstart, outgoingstop);
                return con;
            }
        }
};    


%feature("notabstract") LinearConnection;
class LinearConnection : public Connection, public Parametrized
{
    public: 
        
        LinearConnection(Module* incoming_p, Module* outgoing_p);
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~LinearConnection();
        
        %extend
        {
            LinearConnection(Module* incoming_p, Module* outgoing_p,
                           double* parameters_p, int parameter_size,
                           double* derivatives_p, int derivative_size,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop)
            {
                int required_size = incomingstop - incomingstart;
                if (outgoingstop - outgoingstart != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Slice sizes are not equal. (%d, %d).",
                         required_size, outgoingstop - outgoingstart);
                    return 0;
                }
                if (parameter_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Parameters have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                if (derivative_size != incomingstop - incomingstart)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Derivatives have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                
                LinearConnection* con = new LinearConnection(
                                            incoming_p, outgoing_p, 
                                            parameters_p, derivatives_p,
                                            incomingstart, incomingstop,
                                            outgoingstart, outgoingstop);
                return con;
            }
        }
};    


%feature("notabstract") BlockPermutationConnection;
class BlockPermutationConnection : public Connection
{
    public:
        BlockPermutationConnection(Module* incoming_p, Module* outgoing_p, 
                                   std::vector<int> sequence_shape,
                                   std::vector<int> block_shape);
        virtual ~BlockPermutationConnection();
        
        std::vector<int>& permutation();
        void invert();
};


%feature("notabstract") PermutationConnection;
class PermutationConnection : public Connection
{
    public:
        PermutationConnection(Module* incoming_p, Module* outgoing_p, 
                              std::vector<int> permutation);
        virtual ~PermutationConnection();
        
        std::vector<int>& permutation();
        void invert();
};


%feature("notabstract") ConvolveConnection;
class ConvolveConnection : public Connection, public Parametrized
{
    public:
        ConvolveConnection(Module* incoming_p, Module* outgoing_p, 
                              int inchunk, int outchunk);
        ~ConvolveConnection();
};

%feature("notabstract") InConvolveConnection;
class InConvolveConnection : public Connection, public Parametrized
{
    public:
        InConvolveConnection(Module* incoming_p, Module* outgoing_p, int chunk);
        ~InConvolveConnection();
};


%feature("notabstract") OutConvolveConnection;
class OutConvolveConnection : public Connection, public Parametrized
{
    public:
        OutConvolveConnection(Module* incoming_p, Module* outgoing_p, int chunk);
        ~OutConvolveConnection();
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
        virtual void clear_derivatives();
        void add_module(Module* module_p, ModuleType type=Simple);
        void add_connection(Connection* con_p);
        virtual void sort();

};        
        
        
class BaseMdrnn : public BaseNetwork {};


%feature("notabstract") SigmoidMdrnn;
class SigmoidMdrnn : public BaseMdrnn
{
    public:
        SigmoidMdrnn(int timedim, int hiddensize);
        ~SigmoidMdrnn();
        
        // TODO: remove this; sorting should be implicit, but does not work for
        // mdrnns somehow.
        virtual void sort();
        
        void set_sequence_shape(int dim, int val);
        int get_sequence_shape(int dim);
        int sequencelength();
        void set_block_shape(int dim, int val);
        int get_block_shape(int dim);
};


%feature("notabstract") TanhMdrnn;
class TanhMdrnn : public BaseMdrnn
{
    public:
        TanhMdrnn(int timedim, int hiddensize);
        ~TanhMdrnn();
        
        // TODO: remove this; sorting should be implicit, but does not work for
        // mdrnns somehow.
        virtual void sort();
        
        void set_sequence_shape(int dim, int val);
        int get_sequence_shape(int dim);
        int sequencelength();
        void set_block_shape(int dim, int val);
        int get_block_shape(int dim);
};


%feature("notabstract") LinearMdrnn;
class LinearMdrnn : public BaseMdrnn
{
    public:
        LinearMdrnn(int timedim, int hiddensize);
        ~LinearMdrnn();
        
        // TODO: remove this; sorting should be implicit, but does not work for
        // mdrnns somehow.
        virtual void sort();
        
        void set_sequence_shape(int dim, int val);
        int get_sequence_shape(int dim);
        int sequencelength();
        void set_block_shape(int dim, int val);
        int get_block_shape(int dim);
};


%feature("notabstract") MdlstmMdrnn;
class MdlstmMdrnn : public BaseMdrnn
{
    public:
         MdlstmMdrnn(int timedim, int hiddensize);
        ~MdlstmMdrnn();
        
        // TODO: remove this; sorting should be implicit, but does not work for
        // mdrnns somehow.
        virtual void sort();
        
        void set_sequence_shape(int dim, int val);
        int get_sequence_shape(int dim);
        int sequencelength();
        void set_block_shape(int dim, int val);
        int get_block_shape(int dim);
};

        
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* sample_p, int samplelength), 
                                           (double* target_p, int targetlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* importance_p, int importancelength)};


class SupervisedSimpleDataset
{
   public:
       SupervisedSimpleDataset(int samplesize, int targetsize);
       ~SupervisedSimpleDataset();
       
       virtual int size();
       int samplesize();
       int targetsize();

       bool has_importance();

};


%extend SupervisedSimpleDataset
{
    void append(double* sample_p, int samplelength, 
                double* target_p, int targetlength)
    {
        if (samplelength != $self->samplesize() or \
            targetlength != $self->targetsize()) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         samplelength, targetlength);
            return;
        }
        $self->append(sample_p, target_p);
    }

    void set_importance(int index, double* importance_p, int importancelength)
    {
        if (importancelength != $self->targetsize()) 
        {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths %d given",
                         importancelength);
            return;
        }
        $self->set_importance(index, importance_p);
    }
    
    PyObject* sample(int index)
    {
        SupervisedSimpleDataset& ds = *($self);
        return PyArray_1DFromDoublePointer($self->samplesize(), ds[index].first);        
    }
    
    PyObject* target(int index)
    {
        SupervisedSimpleDataset& ds = *($self);
        return PyArray_1DFromDoublePointer($self->targetsize(), ds[index].second);
    }
    
    PyObject* importance(int index)
    {
        SupervisedSimpleDataset& ds = *($self);
        return PyArray_1DFromDoublePointer($self->targetsize(), ds.importance(index));
    }
};


%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2)
{
    (double* sequence_p, int samplelength, int sequencelength)
};


class SupervisedSemiSequentialDataset
{
   public:
       SupervisedSemiSequentialDataset(int samplesize, int targetsize);
       ~SupervisedSemiSequentialDataset();
       
       virtual int size();
       int samplesize();
       int targetsize();
};


%extend SupervisedSemiSequentialDataset
{
    void append(double* sequence_p, int samplelength, int sequencelength, 
                double* target_p, int targetlength)
    {
        if (samplelength != $self->samplesize() or \
            targetlength != $self->targetsize()) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d, %d) given",
                         samplelength, targetlength);
            return;
        }
        Sequence seq(sequencelength, samplelength, sequence_p);
        $self->append(seq, target_p);
    }
    
    PyObject* sample(int index)
    {
        SupervisedSemiSequentialDataset& ds = *($self);
        Sequence& seq = ds[index].first;
        return PyArray_2DFromDoublePointer(seq.itemsize(), seq.length(), seq[0]);
    }
    
    PyObject* target(int index)
    {
        SupervisedSemiSequentialDataset& ds = *($self);
        return PyArray_1DFromDoublePointer($self->targetsize(), ds[index].second);
    }
};


%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* samplesequence_p, int samplelength, int samplesequencelength), 
    (double* targetsequence_p, int targetlength, int targetsequencelength)
};


class SupervisedSequentialDataset
{
   public:
       SupervisedSequentialDataset(int samplesize, int targetsize);
       ~SupervisedSequentialDataset();
       
       virtual int size();
       int samplesize();
       int targetsize();
};


%extend SupervisedSequentialDataset
{
    void append(double* samplesequence_p, int samplelength, int samplesequencelength, 
                double* targetsequence_p, int targetlength, int targetsequencelength)
    {
        if (samplelength != $self->samplesize() or targetlength != $self->targetsize()) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         samplelength, targetlength);
            return;
        }
        if (samplesequencelength != targetsequencelength) {
            PyErr_Format(PyExc_ValueError, 
                "Sequences have to be of same length.");
            return;
        }
        
        Sequence sampleseq(samplesequencelength, samplelength, samplesequence_p);
        Sequence targetseq(targetsequencelength, targetlength, targetsequence_p);
        $self->append(sampleseq, targetseq);
    }
    
    PyObject* sample(int index)
    {
        SupervisedSequentialDataset& ds = *($self);
        Sequence& seq = ds[index].first;
        return PyArray_2DFromDoublePointer(seq.itemsize(), seq.length(), seq[0]);
    }

    PyObject* target(int index)
    {
        SupervisedSequentialDataset& ds = *($self);
        Sequence& seq = ds[index].second;
        return PyArray_2DFromDoublePointer(seq.itemsize(), seq.length(), seq[0]);
    }
};

%clear (double* INPLACE_ARRAY2, int DIM1, int DIM2);

class SimpleBackprop
{
    public:
        SimpleBackprop(BaseNetwork& network, SupervisedSimpleDataset& ds);
        virtual ~SimpleBackprop();
    
        void train_stochastic();    
        void train_stochastic_batch();    

        double loss();
        
        double learningrate();
        void set_learningrate(const double value);
        
        double momentum();
        void set_momentum(const double value);
        
        BaseNetwork& network();
        SupervisedSimpleDataset& dataset();
};


%extend SimpleBackprop
{
    PyObject* error() 
    {
        return PyArray_1DFromDoublePointer($self->network().outsize(), 
                                           $self->error());
    }
}


class SemiSequentialBackprop
{
    public:

        SemiSequentialBackprop(BaseNetwork& network, 
                               SupervisedSemiSequentialDataset& ds);
        ~SemiSequentialBackprop();
    
        void train_stochastic();    
        
        double learningrate();
        void set_learningrate(const double value);
        
        double momentum();
        void set_momentum(const double value);
        
        BaseNetwork& network();
        SupervisedSemiSequentialDataset& dataset();
};


%extend SemiSequentialBackprop
{
    PyObject* lasterror() 
    {
        return PyArray_1DFromDoublePointer($self->dataset().targetsize(), 
                                           $self->error());
    }
}


class SequentialBackprop
{
    public:
        SequentialBackprop(BaseNetwork& network, SupervisedSequentialDataset& ds);
        ~SequentialBackprop();
    
        void train_stochastic();
        
        double learningrate();
        void set_learningrate(const double value);
        
        double momentum();
        void set_momentum(const double value);
        
        BaseNetwork& network();
        SupervisedSequentialDataset& dataset();
};


double gradient_check(BaseNetwork& network);
