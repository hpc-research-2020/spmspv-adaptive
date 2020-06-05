#!/bin/bash
####used for test the adaptiveness or application.

#EXEC=../../build/bin/my_spmspv_adaptive_test
#EXEC=../../build/bin/my_spmspv_application_test
#EXEC=../../build/bin/test_vector_format_conversion
#EXEC=../../build/bin/test_holaspmv
EXEC=../../build/bin/generate_model_data

DATASET=../../data/matrix-market/ipdpsw/*.mtx
$EXEC $DATASET

