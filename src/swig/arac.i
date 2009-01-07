%module arac
%{
#include "../cpp/arac.h"
#include "../cpp/structure/component.h"
#include "../cpp/structure/parametrized.h"
#include "../cpp/structure/connections/connection.h"
#include "../cpp/structure/modules/module.h"

using arac::structure::Component;
using arac::structure::Parametrized;
using arac::structure::connections::Connection;
using arac::structure::modules::Module;

%}

%import "../cpp/structure/modules/module.h"
%import "../cpp/structure/component.h"

%include "../cpp/structure/modules/bias.h"
%include "../cpp/structure/modules/gate.h"
%include "../cpp/structure/modules/linear.h"
%include "../cpp/structure/modules/mdlstm.h"
%include "../cpp/structure/modules/lstm.h"
%include "../cpp/structure/modules/partialsoftmax.h"
%include "../cpp/structure/modules/sigmoid.h"
%include "../cpp/structure/modules/softmax.h"
%include "../cpp/structure/modules/tanh.h"

%include "../cpp/structure/connections/identity.h"
%include "../cpp/structure/connections/full.h"
%include "../cpp/structure/connections/linear.h"

%include "../cpp/structure/networks/network.h"

