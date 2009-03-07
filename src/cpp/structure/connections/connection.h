// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_CONNECTION_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_CONNECTION_INCLUDED


#include <cassert>

#include "../component.h"
#include "../modules/module.h"



namespace arac {
namespace structure {
namespace connections {
    
    
using namespace arac::structure::modules;

/// 
/// A connection connects two modules by taking the ouput of the incoming
/// module, transforming it and adding it to the input of the outgoing module.
///
/// It is possible to connect only parts of two modules. For example, one could
/// connect the second half of the incoming module to the first half of the 
/// outgoing module.
///

class Connection : public arac::structure::Component
{
    public: 
        
        ///
        /// Create a new connection object. The exact slices of the connections
        /// can be specified by setting the indices; the default is to connect
        /// over the full length.
        /// 
        Connection(Module* incoming, Module* outgoing,
                   int incomingstart, int incomingstop, 
                   int outgoingstart, int outgoingstop);
        Connection(Module* incoming, Module* outgoing);
        virtual ~Connection();
        
        ///
        /// Set the incoming start index.
        ///
        void set_incomingstart(int n);
        
        ///
        /// Set the incoming stop index.
        ///
        void set_incomingstop(int n);
        
        ///
        /// Set the outgoing start index.
        ///
        void set_outgoingstart(int n);
        
        ///
        /// Set the outgoing stop index.
        ///
        void set_outgoingstop(int n);
        
        ///
        /// Return the incoming start index.
        ///
        int get_incomingstart();
        
        ///
        /// Return the incoming stop index.
        ///
        int get_incomingstop();
        
        ///
        /// Return the outgoing start index.
        ///
        int get_outgoingstart();
        
        ///
        /// Return the outgoing stop index.
        ///
        int get_outgoingstop();
        
        ///
        /// Set the grade of recurrency.
        ///
        void set_recurrent(int recurrent);
        
        ///
        /// Return the grade of recurrency.
        ///
        int get_recurrent();

        ///
        /// Return a pointer to the incoming module.
        ///
        Module* incoming();
        
        ///
        /// Return a pointer to the outgoing module.
        ///
        Module* outgoing();
        
    protected:
    
        void _forward();
        void _backward();

        virtual void forward_process(double* sink_p, const double* source_p) = 0;
        virtual void backward_process(double* sink_p, const double* source_p) = 0;
        
        Module* _incoming_p;
        Module* _outgoing_p;
        
        int _incomingstart;
        int _incomingstop;
        int _outgoingstart;
        int _outgoingstop;
        
        int _recurrent;
};
    
    
inline
void
Connection::set_incomingstart(int n)
{
    _incomingstart = n;
    
}


inline
void
Connection::set_incomingstop(int n)
{
    _incomingstop = n;
}


inline
void
Connection::set_outgoingstart(int n)
{
    _outgoingstart = n;
}


inline
void
Connection::set_outgoingstop(int n)
{
    _outgoingstop = n;
}


inline
int
Connection::get_incomingstart()
{
    return _incomingstart;
}


inline
int
Connection::get_incomingstop()
{
    return _incomingstop;
}


inline
int
Connection::get_outgoingstart()
{
    return _outgoingstart;
    
}


inline
int
Connection::get_outgoingstop()
{
    return _outgoingstop;
}


inline
int
Connection::get_recurrent()
{
    return _recurrent;
}


inline
void
Connection::set_recurrent(int recurrent)
{
    assert((!recurrent) || (sequential()));
    _recurrent = recurrent;
}


inline
Module* 
Connection::incoming()
{
    return _incoming_p;
}


inline
Module* 
Connection::outgoing()
{
    return _outgoing_p;
}

    
}
}
}


#endif
