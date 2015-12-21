%module falconn
%{
#define SWIG_FILE_WITH_INIT
#include "../include/falconn/falconn_global.h"
#include "python/python_wrapper.h"

using falconn::LSHConstructionParameters;
%}

%include <std_vector.i>
%include <stdint.i>
%include "numpy.i"

%init %{
import_array();
%}

namespace std {
  %template(ResultList) vector<int32_t>;
}

%apply (double* IN_ARRAY1, int DIM1) {(const double* vec, int len)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* matrix, int num_rows, int num_columns)};

%include "python_wrapper.h"
