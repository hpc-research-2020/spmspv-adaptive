#!/bin/bash
#EXEC=../../build/bin/csr5_spmspv_test
EXEC=../../build/bin/test_holaspmv

DATASET=../../data/*.mtx
$EXEC $DATASET
