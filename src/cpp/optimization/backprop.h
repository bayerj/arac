// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_OPTIMIZER_BACKPROP_INCLUDED
#define Arac_OPTIMIZER_BACKPROP_INCLUDED


#include <iostream>

#include "../structure/networks/network.h"
#include "../datasets/datasets.h"


using arac::structure::networks::Network;
using arac::structure::Parametrized;
using arac::datasets::SupervisedDataset;
using arac::datasets::Sequence;


namespace arac {
namespace optimization {
    
    
// TODO: document.
template<typename SampleType, typename TargetType>
class Backprop
{
    public: 
    
        typedef SupervisedDataset<SampleType, TargetType> DatasetType;
    
        Backprop(Network& network, DatasetType& dataset);
        virtual ~Backprop();
        
        Network& network();
        
        DatasetType& dataset();
        
        const double& learningrate();
        
        void set_learningrate(const double value);

        void train_stochastic();
        
    protected:
        
        void learn();
        
        Network& _network;
        DatasetType& _dataset;
        double _learningrate;
        double* _error_p;

        // FIXME: this function should be abstract instead. But in that case,
        // classes inheriting from this class which give a definition for the
        // concrete class are still abstract somehow. wtf...?
        virtual void process_sample(const SampleType inpt, 
                                    const TargetType target) {
                                        std::cout << "Don't use!" << std::endl;
                                    };

};


template<typename SampleType, typename TargetType>
Backprop<SampleType, TargetType>::Backprop(Network& network, 
                                            DatasetType& dataset) :
    _network(network),
    _dataset(dataset),
    _learningrate(0.001)
{
    _network.sort();
    assert(_network.insize() == _dataset.samplesize());
    assert(_network.outsize() == _dataset.targetsize());
    _error_p = new double[_network.outsize()];
}


template<typename SampleType, typename TargetType>
Backprop<SampleType, TargetType>::~Backprop()
{
    delete[] _error_p;
}


template<typename SampleType, typename TargetType>
Network&
Backprop<SampleType, TargetType>::network()
{
    return _network;
}


template<typename SampleType, typename TargetType>
SupervisedDataset<SampleType, TargetType>&
Backprop<SampleType, TargetType>::dataset()
{
    return _dataset;
}
 
 
template<typename SampleType, typename TargetType>
const double&
Backprop<SampleType, TargetType>::learningrate()
{
    return _learningrate;
}
 
 
template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::set_learningrate(const double value)
{
    _learningrate = value;
}


template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::train_stochastic()
{
    int index = rand() % dataset().size();
    
    SampleType sample = dataset()[index].first;
    TargetType target = dataset()[index].second;
    network().clear();
    this->process_sample(sample, target);
    learn();
}


template<typename SampleType, typename TargetType>
void 
Backprop<SampleType, TargetType>::learn()
{
    std::vector<Parametrized*>::const_iterator param_iter;
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         param_iter++)
    {
        double* params_p = (*param_iter)->get_parameters();
        double* derivs_p = (*param_iter)->get_derivatives();
        for (int i = 0; i < (*param_iter)->size(); i++)
        {
            params_p[i] += _learningrate * derivs_p[i];
        }
    }
}


//
// Spezializations.
//
 
class SimpleBackprop : public Backprop<double*, double*> 
{
    public:
        SimpleBackprop(Network& network, 
                       SupervisedDataset<double*, double*>& dataset);
        ~SimpleBackprop();
    
    protected:
        virtual void process_sample(const double* input, 
                                    const double* target)
        {
            std::cout << "processing for double* double*" << std::endl;
            const double* output_p = network().activate(input_p);
            for (int i = 0; i < network().outsize(); i++)
            {
                _error_p[i] = target_p[i] - output_p[i];
            }
            network().back_activate(_error_p);
        }
    
};


class SemiSequentialBackprop : public Backprop<Sequence, double*>
{
    
    public:
        SemiSequentialBackprop(Network& network,
                               SupervisedDataset<Sequence, double*>& dataset);
        ~SemiSequentialBackprop();

    protected:
        virtual void process_sample(const Sequence input_p, 
                                    const double* target_p)
        {
            // FIXME: implement
        }
};


class SequentialBackprop : public Backprop<Sequence, Sequence>
{
    public:
        SequentialBackprop(Network& network, 
                           SupervisedDataset<double*, double*>& dataset);
        ~SequentialBackprop();

    protected:
        virtual void process_sample(const Sequence input, 
                                    const Sequence target)
        {
            // FIXME: implement
        }
};

 
}
}

#endif