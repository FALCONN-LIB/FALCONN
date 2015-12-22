%module falconn
%{
#define SWIG_FILE_WITH_INIT
#include <exception>
#include <string>
#include "../../include/falconn/falconn_global.h"
#include "../../include/falconn/lsh_nn_table.h"
#include "python/module/python_wrapper.h"

using falconn::LSHConstructionParameters;
using falconn::FalconnError;
%}

%include <exception.i>
%include <std_string.i>
%include <std_vector.i>
%include <stdint.i>
%include "numpy.i"

%init %{
import_array();
%}

namespace std {
  %template(ResultList) vector<int32_t>;
}

%ignore python_to_cpp_construction_parameters;

%exception {
  try {
    $action
  } catch (const falconn::FalconnError& e) {
    std::string s1("FALCONN error: ");
    // TODO: add more type information
    std::string s2(e.what());
    std::string msg = s1 + s2;
    SWIG_exception(SWIG_RuntimeError, msg.c_str());
  } catch (const std::exception& e) {
    std::string s1("std::exception error: ");
    std::string s2(e.what());
    std::string msg = s1 + s2;
    SWIG_exception(SWIG_RuntimeError, msg.c_str());
  } catch (...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

%apply (double* IN_ARRAY1, int DIM1) {(const double* vec, int len)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* matrix, int num_rows, int num_columns)};

%include "python_wrapper.h"
