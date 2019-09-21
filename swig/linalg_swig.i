/* -*- c++ -*- */

#define LINALG_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "linalg_swig_doc.i"

%{
#include "linalg/multiply_cc.h"
#include "linalg/stream_to_matrix_cc.h"
#include "linalg/matrix_to_stream_cc.h"
#include "linalg/multiply_cc.h"
#include "linalg/stream_to_matrix_cc.h"
#include "linalg/matrix_to_stream_cc.h"
%}

%include "linalg/multiply_cc.h"
GR_SWIG_BLOCK_MAGIC2(linalg, multiply_cc);
%include "linalg/multiply_cc.h"
GR_SWIG_BLOCK_MAGIC2(linalg, multiply_cc);
%include "linalg/stream_to_matrix_cc.h"
GR_SWIG_BLOCK_MAGIC2(linalg, stream_to_matrix_cc);
%include "linalg/matrix_to_stream_cc.h"
GR_SWIG_BLOCK_MAGIC2(linalg, matrix_to_stream_cc);
