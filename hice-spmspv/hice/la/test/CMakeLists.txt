set( mgpu_SRC_FILES "${EXTERNAL_PROJECT_DIR}/moderngpu/src/src/mgpucontext.cu" "${EXTERNAL_PROJECT_DIR}/moderngpu/src/src/mgpuutil.cpp")
#################statistic test.
add_executable(statistic statistic.cu)
target_compile_definitions(statistic
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(statistic
  PROPERTIES
  LINK_FLAGS -fopenmp)
target_link_libraries(statistic la cudart cusparse)


#################generate features.
add_executable(generate_feature generate_feature.cu)
target_compile_definitions(generate_feature
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(generate_feature
  PROPERTIES
  LINK_FLAGS -fopenmp)
target_link_libraries(generate_feature la cudart cusparse)

#################generate model trainning data for class 3.
add_executable(generate_model_data_model3 "generate_model_data_model3.cu" ${mgpu_SRC_FILES})
target_compile_definitions(generate_model_data_model3
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
  #DOUBLE)
set_target_properties(generate_model_data_model3
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(generate_model_data_model3 la cudart cusparse)

#################generate model trainning data.
add_executable(generate_model_data "generate_model_data.cu" ${mgpu_SRC_FILES})
target_compile_definitions(generate_model_data
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
  #DOUBLE)
set_target_properties(generate_model_data
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(generate_model_data la cudart cusparse)

################test vector format conversion's correctness.
add_executable(test_vector_format_conversion test_vector_format_conversion.cu)
target_compile_definitions(test_vector_format_conversion
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(test_vector_format_conversion
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(test_vector_format_conversion la cudart cusparse)

################for all of the tests of my lib.
#add_executable(new_spmspv_test new_spmspv_test.cu)
#target_compile_definitions(new_spmspv_test
#  PUBLIC
#  #VALUE_TYPE=double
#    VALUE_TYPE=float
#    NUM_RUN=10
#    #DOUBLE
#)
#set_target_properties(new_spmspv_test
#  PROPERTIES
#  #-Xcompiler -fopenmp)
#  LINK_FLAGS -fopenmp)
#target_link_libraries(new_spmspv_test la cudart cusparse)

################test for bfs (paper)
add_executable(test_application_model3 "test_application_model3.cu" ${mgpu_SRC_FILES})
target_compile_definitions(test_application_model3
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(test_application_model3
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(test_application_model3 la cudart cusparse)

################test for bfs (paper)
add_executable(test_application "test_application.cu" ${mgpu_SRC_FILES})
target_compile_definitions(test_application
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(test_application
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(test_application la cudart cusparse)

################adaptive framework perf(paper)
add_executable(new_test_app "new_test_app.cu" ${mgpu_SRC_FILES})
target_compile_definitions(new_test_app
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
    #DOUBLE)
set_target_properties(new_test_app
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(new_test_app la cudart cusparse)

#################only for my method with different x_nnz. (adaptive)
add_executable(my_spmspv_adaptive_test "my_spmspv_adaptive_test.cu" ${mgpu_SRC_FILES})
target_compile_definitions(my_spmspv_adaptive_test
  PUBLIC
  VALUE_TYPE=float)
  #DOUBLE)
set_target_properties(my_spmspv_adaptive_test
  PROPERTIES
  LINK_FLAGS -fopenmp)
target_link_libraries(my_spmspv_adaptive_test la cudart cusparse)

# ################only for my method in application.
# add_executable(my_spmspv_application_test my_spmspv_application_test.cu)
# target_compile_definitions(my_spmspv_application_test
#   PUBLIC
#     VALUE_TYPE=float
#     NUM_RUN=10)
#     #DOUBLE)
# set_target_properties(my_spmspv_application_test
#   PROPERTIES
#   #-Xcompiler -fopenmp)
#   LINK_FLAGS -fopenmp)
# target_link_libraries(my_spmspv_application_test la cudart cusparse)

##########only for csr5 spmspv with different x_nnz.##########
add_executable(csr5_spmspv_test csr5_spmspv_test.cu)
target_compile_definitions(csr5_spmspv_test
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
  #DOUBLE)
set_target_properties(csr5_spmspv_test
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(csr5_spmspv_test la cudart cusparse)

##########holaspmv and holaspmspv test.##########
add_executable(test_holaspmv test_holaspmv.cu)
target_compile_definitions(test_holaspmv
  PUBLIC
    VALUE_TYPE=float
    NUM_RUN=10)
  #DOUBLE)
set_target_properties(test_holaspmv
  PROPERTIES
  #-Xcompiler -fopenmp)
  LINK_FLAGS -fopenmp)
target_link_libraries(test_holaspmv la cudart cusparse)


# add_executable(my_spmspv_test my_spmspv_test.cu)
# target_compile_definitions(my_spmspv_test
#   PUBLIC
#     VALUE_TYPE=float
#     NUM_RUN=10
#     #DOUBLE
# )
# set_target_properties(my_spmspv_test
#   PROPERTIES
#   #-Xcompiler -fopenmp)
#   LINK_FLAGS -fopenmp)
# target_link_libraries(my_spmspv_test la cudart cusparse)

# add_executable(bucket_spmspv_test bucket_spmspv_test.cu)
# target_compile_definitions(bucket_spmspv_test
#   PUBLIC
#     VALUE_TYPE=float
#     NUM_RUN=10
#     #DOUBLE
# )
# set_target_properties(bucket_spmspv_test
#   PROPERTIES
#   #-Xcompiler -fopenmp)
#   LINK_FLAGS -fopenmp)
# target_link_libraries(bucket_spmspv_test la cudart cusparse)

############test sort-spmspv##################################
# add_executable(test_sort_spmspv test_sort_spmspv.cu)
# target_compile_definitions(test_sort_spmspv
#   PUBLIC
#     VALUE_TYPE=float
#     NUM_RUN=10
#     #DOUBLE
# )
# set_target_properties(test_sort_spmspv
#   PROPERTIES
#   #-Xcompiler -fopenmp)
#   LINK_FLAGS -fopenmp)
# target_link_libraries(test_sort_spmspv la cudart cusparse)

##########test csr2csc kerenl's perf.##########
# add_executable(test_csr2csc test_csr2csc.cu)
# target_compile_definitions(test_csr2csc
#   PUBLIC
#     VALUE_TYPE=float
#     NUM_RUN=10)
#   #DOUBLE)
# set_target_properties(test_csr2csc
#   PROPERTIES
#   #-Xcompiler -fopenmp)
#   LINK_FLAGS -fopenmp)
# target_link_libraries(test_csr2csc la cudart cusparse)

#########test coo to hybrid format##########
 add_executable(test_coo2hybrid test_coo2hybrid.cu)
 target_compile_definitions(test_coo2hybrid
   PUBLIC
     VALUE_TYPE=float
     NUM_RUN=10)
   #DOUBLE)
 set_target_properties(test_coo2hybrid
   PROPERTIES
   #-Xcompiler -fopenmp)
   LINK_FLAGS -fopenmp)
 target_link_libraries(test_coo2hybrid la cudart cusparse)
