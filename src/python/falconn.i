%module falconn
%{
#define SWIG_FILE_WITH_INIT
#include "../include/falconn/falconn_global.h"
#include "python/python_wrapper.h"
%}

%include <std_vector.i>
%include <stdint.i>
%include "numpy.i"

%init %{
import_array();
%}

namespace std {
  %template(ResultList) vector<int_fast64_t>;
}

%apply (double* IN_ARRAY1, int DIM1) {(const double* vec, int len)};

%include "python_wrapper.h"
