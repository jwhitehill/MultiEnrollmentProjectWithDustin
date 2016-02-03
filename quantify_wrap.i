%module quantify

%{
#include "quantify.hpp"
%}

// Allow Swig to convert from Python buffer to float
%include <pybuffer.i>
%pybuffer_string(float *);
%typemap(typecheck) float* = PyObject *;

%include "quantify.hpp"
