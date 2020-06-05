#!/bin/bash
####used for test application and feature's overheads

#EXEC=../../build/bin/test_application
#EXEC=../../build/bin/test_vector_format_conversion
#EXEC=../../build/bin/test_holaspmv
#EXEC=../../build/bin/my_spmspv_adaptive_test
#EXEC=../../build/bin/test_application_model3
EXEC=../../build/bin/new_test_app
#EXEC=../../build/bin/test_coo2hybrid

DATA=

DATASET=$DATA/delaunay_n13.mtx
$EXEC $DATASET

