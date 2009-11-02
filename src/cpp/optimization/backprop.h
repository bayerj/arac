// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_OPTIMIZER_BACKPROP_INCLUDED
#define Arac_OPTIMIZER_BACKPROP_INCLUDED


#include <iostream>
#include <cstdlib>

#include "../structure/networks/network.h"
#include "../datasets/datasets.h"
#include "descent/stepdescender.h"
#include "descent/descender.h"


using arac::structure::networks::BaseNetwork;
using arac::structure::Parametrized;
using arac::datasets::SupervisedDataset;
using arac::datasets::Sequence;
using arac::optimization::descent::Descender;
using arac::optimization::descent::StepDescender;


namespace arac {
namespace optimization {
    
    
///
/// Template class for Backprop optimizers.
///
/// Each optimizer holds its own implementation of how to process a sample.
///

template<typename SampleType, typename TargetType>
class Backprop
{
    public: 
        
        ///
        /// Backprop implementaion specific dataset type on which a network
        /// is to be trained.
        ///
        typedef SupervisedDataset<SampleType, TargetType> DatasetType;
    
        ///
        /// Create a new Backprop optimizer object with which the given network
        /// can be trained on the given dataset.
        ///
        Backprop(BaseNetwork& network, DatasetType& dataset);
        
        ///
        /// Destroy the Backprop object.
        ///
        virtual ~Backprop();
        
        ///
        /// Return a reference to the network of the optimizer.
        ///
        BaseNetwork& network();
        
        ///
        /// Return a reference to the dataset of the optimizer.
        ///
        DatasetType& dataset();

        ///
        /// Return the learningrate of the trainer.
        ///
        double learningrate();
       
        ///
        /// Set the learningrate of the trainer.
        ///
        void set_learningrate(const double value);        

        ///
        /// Return the momentum of the trainer.
        ///
        double momentum();
       
        ///
        /// Set the momentum of the trainer.
        ///
        void set_momentum(const double value);        

        ///
        /// Return a pointer to the last error vector.
        ///
        double* error();

        ///
        /// Return a pointer to the last loss.
        ///
        double loss();
       
        ///
        /// Pick a random sample from the dataset and perform one step of
        /// backpropagation.
        ///
        /// This method calls the methods process_sample and learn, of which the
        /// former is to be implemented by subclasses.
        ///
        void train_stochastic();

        ///
        /// Perform online training on the dataset. Samples are drawn without
        /// replacemet.
        ///
        /// This method calls the methods process_sample and learn, of which the
        /// former is to be implemented by subclasses.
        ///
        void train_stochastic_batch();
        
    protected:
        
        ///
        /// Adapt the parameters of this network.
        ///
        void learn();

        ///
        /// Network to be optimized optimizer.
        ///
        BaseNetwork& _network;
        
        ///
        /// Dataset the network is to be optimized upon.
        ///
        DatasetType& _dataset;
        
        ///
        /// Descender object that is used to follow the gradient.
        ///
        // TODO: this should be of the abstract Descender class.
        StepDescender _descender;
        
        ///
        /// The last error during process_sample.
        ///
        double* _error_p;
        
        ///
        /// Method that processes a sample from the dataset and its target. This
        /// method has to be implemented for each (SampleType, TargetType) 
        /// combination.
        ///
        virtual void process_sample(SampleType inpt, TargetType target)  = 0;
            
        ///
        /// Method that processes a sample from the dataset and its target, 
        /// additionally scaling the error at the targets according to an 
        /// importance.
        /// 
        virtual void process_sample(SampleType inpt, TargetType target, 
                                    TargetType importance)  = 0;

    private:

        double _loss;

};


template<typename SampleType, typename TargetType>
Backprop<SampleType, TargetType>::Backprop(BaseNetwork& network, 
                                           DatasetType& dataset) :
    _network(network),
    _dataset(dataset),
    _descender(network, 0.005),
    _loss(0.0)
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
BaseNetwork&
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
double
Backprop<SampleType, TargetType>::learningrate()
{
    return _descender.stepratio();
}


template<typename SampleType, typename TargetType>
double
Backprop<SampleType, TargetType>::loss()
{
    return _loss;
}


template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::set_learningrate(const double value)
{
    _descender.set_stepratio(value);
}


template<typename SampleType, typename TargetType>
double
Backprop<SampleType, TargetType>::momentum()
{
    return _descender.momentum();
}


template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::set_momentum(const double value)
{
    _descender.set_momentum(value);
}


template<typename SampleType, typename TargetType>
double*
Backprop<SampleType, TargetType>::error()
{
    return _error_p;
}


template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::train_stochastic()
{
    int index = rand() % dataset().size();
    
    network().clear();
    network().clear_derivatives();
    SampleType sample = dataset()[index].first;
    TargetType target = dataset()[index].second;
    if (dataset().has_importance())
    {
        this->process_sample(sample, target, dataset().importance(index));
    }
    else
    {
        this->process_sample(sample, target);
    }
    learn();
}


template<typename SampleType, typename TargetType>
void
Backprop<SampleType, TargetType>::train_stochastic_batch()
{
    _loss = 0.0;

    // Create a vector of indices and shuffle it.
    std::vector<int> indices;
    indices.reserve(dataset().size());
    for (int i = 0; i < dataset().size(); ++i)
    {
        indices.push_back(i);
    }
    for (int i = 0; i < dataset().size(); ++i)
    {
        int rest = dataset().size() - i;
        int index = rand() % rest;
        int swap = indices[i];
        indices[i] = indices[i + index];
        indices[i + index] = swap;
    }

    std::vector<int>::const_iterator index_iter;
    for (index_iter = indices.begin(); 
         index_iter != indices.end();
         ++index_iter)
    {
        int index = *index_iter;
        network().clear();
        network().clear_derivatives();
    
        SampleType sample = dataset()[index].first;
        TargetType target = dataset()[index].second;
        if (dataset().has_importance())
        {
            this->process_sample(sample, target, dataset().importance(index));
        }
        else
        {
            this->process_sample(sample, target);
        }
        learn();
 
        // Update loss.
        for (int i = 0; i < network().outsize(); ++i)
        {
            _loss += _error_p[i] * _error_p[i];
        }
    }
}


template<typename SampleType, typename TargetType>
void 
Backprop<SampleType, TargetType>::learn()
{
    _descender.notify();
}


//
// Spezializations.
//


///
/// SimpleBackprop optimizers are used to train a feedforward network on a 
/// dataset that consists of independet sample and target vectors.
///
 
class SimpleBackprop : public Backprop<double*, double*> 
{
    public:
        SimpleBackprop(BaseNetwork& network, 
                       SupervisedDataset<double*, double*>& dataset);
        ~SimpleBackprop();
    
    protected:
        virtual void process_sample(double* input_p, double* target_p);
        virtual void process_sample(double* input_p, double* target_p,
                                    double* importance_p);
};


///
/// SemiSequentialBackprop optimizers are used to train networks that map 
/// sequences to vectors.
///

class SemiSequentialBackprop : public Backprop<Sequence, double*>
{
    
    public:
        SemiSequentialBackprop(BaseNetwork& network,
                               SupervisedDataset<Sequence, double*>& dataset);
        ~SemiSequentialBackprop();

    protected:
        virtual void process_sample(Sequence input_p, double* target_p);
        virtual void process_sample(Sequence input_p, double* target_p,
                                    double* importance_p);

    private:
        double* _output_p;
};

///
/// SequentialBackprop optimizers are used for networks that map sequences to
/// sequences of the same length.
///

class SequentialBackprop : public Backprop<Sequence, Sequence>
{
    public:
        SequentialBackprop(BaseNetwork& network, 
                           SupervisedDataset<Sequence, Sequence>& dataset);
        ~SequentialBackprop();

    protected:
        virtual void process_sample(Sequence input, Sequence target);
        virtual void process_sample(Sequence input, Sequence target,
                                    Sequence importance);
        
    private:
        std::vector<const double*> _outputs;

};

 
}
}

#endif
