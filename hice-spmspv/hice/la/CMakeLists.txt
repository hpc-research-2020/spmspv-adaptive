cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(hice-la VERSION 0.0.1 LANGUAGES CXX C Fortran)

##############################################
# Check openmp setting
##############################################

##set(CMAKE_CXX_FLAGS "-std=c++11" )

FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
 SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

##############################################
# Check CUDA setting
##############################################
option(USE_CUDA "Use CUDA" ON)
if(USE_CUDA)
  enable_language(CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  find_package(CUDA QUIET REQUIRED)

  #set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_60,code=sm_60 -Xcompiler=-Wall")
  #set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_60,code=sm_60 -Xcompiler -fopenmp")
  #set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_60,code=sm_60 -Xcompiler -fopenmp --ptxas-options=-v -keep")
  #set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_60,code=sm_60 -Xcompiler -fopenmp")
  set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_70,code=sm_70 -Xcompiler -fopenmp")
  #set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_35,code=sm_35 -Xcompiler -fopenmp")

endif()

##############################################
# Genernal setup
##############################################

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_COLOR_MAKEFILE   ON)

set(ROOT_PROJECT_NAME hice)
set(ROOT_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(EXTERNAL_PROJECT_DIR "${ROOT_PROJECT_DIR}/third_party")
#set(PROJECT_NAME "${ROOT_PROJECT_NAME}_la")

message(STATUS "Project will be installed to ${CMAKE_INSTALL_PREFIX}")

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
#endif()
#
#message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}")

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Offer the user the choice of overriding the installation directories
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake" CACHE PATH "Installation directory for CMake files")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")

# Report to user
foreach(p LIB BIN INCLUDE CMAKE)
  file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_${p}DIR} _path)
  message(STATUS "Installing ${p} to ${_path}")
  unset(_path)
endforeach()

set(STAGED_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage)
message(STATUS "Project staged install: ${STAGED_INSTALL_PREFIX}")

option(USE_SYSTEM_BLAS "Use system BLAS" OFF)
option(USE_SYSTEM_LAPACK "Use system LAPACK" OFF)

##############################################
# Add the external projects
##############################################
if(NOT USE_SYSTEM_BLAS OR NOT USE_SYSTEM_LAPACK)
  #list(APPEND CMAKE_MODULE_PATH "${EXTERNAL_PROJECT_DIR}/lapack/lapack-build")
  #include_directories(${STAGED_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
  #link_directories(${STAGED_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  #include(${ROOT_PROJECT_DIR}/cmake/external/FetchLAPACK.cmake)
  #include(${ROOT_PROJECT_DIR}/cmake/external/ExternalLapack.cmake)
  #set(CBLAS_DIR ${STAGED_INSTALL_PREFIX}/lib64/cmake/cblas-3.8.0)
  #find_package(CBLAS)
endif()

include(${ROOT_PROJECT_DIR}/cmake/external/ExternalCub.cmake)
include(${ROOT_PROJECT_DIR}/cmake/external/ExternalModernGPU.cmake)
include(${ROOT_PROJECT_DIR}/cmake/external/ExternalCSR5.cmake)

##############################################
# Create target and set properties
##############################################

FILE(GLOB_RECURSE la_cpu_srcs "src/*.cpp" "src/*.c" "src/*.h" "include/*.h")
FILE(GLOB_RECURSE la_cuda_srcs "src/*.cu" "src/*.cuh")

add_library(la
  ${la_cpu_srcs}
  ${la_cuda_srcs}
)

set_target_properties(la
  PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON
#    INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_include_directories(la
    PUBLIC
        $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc>
        $<BUILD_INTERFACE:${EXTERNAL_PROJECT_DIR}/csr5/src>
        $<BUILD_INTERFACE:${EXTERNAL_PROJECT_DIR}/cub/src>
        #$<BUILD_INTERFACE:${EXTERNAL_PROJECT_DIR}/moderngpu/src>
        $<BUILD_INTERFACE:${EXTERNAL_PROJECT_DIR}/moderngpu/src/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        #${CBLAS_INCLUDE_DIRS}
)

target_link_libraries(la
  PRIVATE
  #cblas
  #${CBLAS_LIBRARIES}
)

##############################################
# Testing
##############################################

option(ENABLE_TESTING "ENABLE tests" ON)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

# Add an alias for testing
#add_library(${ROOT_PROJECT_NAME}::la ALIAS la)

###############################################
## Installation
###############################################
#install(TARGETS la
#  EXPORT la_targets
#  LIBRARY DESTINATION ${INSTALL_LIBDIR}
#  ARCHIVE DESTINATION ${INSTALL_LIBDIR}
#  RUNTIME DESTINATION ${INSTALL_BINDIR}
#)
#
#install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDEDIR})
#
## Export the targets to a script
#install(EXPORT la_targets
#  FILE
#     HICE_LATargets.cmake
#  NAMESPACE
#    ${ROOT_PROJECT_NAME}::
#  DESTINATION
#    ${INSTALL_CMAKEDIR}
#)
#
#include(CMakePackageConfigHelpers)
#
## Create a Config.cmake file
#configure_package_config_file(
#  ${ROOT_PROJECT_DIR}/cmake/template/HICE_LAConfig.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/HICE_LAConfig.cmake
#  INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
#)
#
## Create a ConfigVersion.cmake file
#write_basic_package_version_file(
#  ${CMAKE_CURRENT_BINARY_DIR}/HICE_LAConfigVersion.cmake
#  VERSION ${PROJECT_VERSION}
#  COMPATIBILITY AnyNewerVersion
#)
#
#install(FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/HICE_LAConfig.cmake
#  ${CMAKE_CURRENT_BINARY_DIR}/HICE_LAConfigVersion.cmake
#  DESTINATION ${INSTALL_CMAKEDIR}
#)
#
#install(
#  DIRECTORY
#  ${STAGED_INSTALL_PREFIX}/
#  DESTINATION
#  .
#  USE_SOURCE_PERMISSIONS
#)

#################################################
### Exporting from the build tree
################################################
##
##configure_file(${ROOT_PROJECT_DIR}/cmake/module/FindBLAS.cmake
##  ${CMAKE_CURRENT_BINARY_DIR}/FindBLAS.cmake
##  COPYONLY
##)
##
##export(EXPORT la_targets
##  FILE ${CMAKE_CURRENT_BINARY_DIR}/HICELATargets.cmake
##  NAMESPACE ${ROOT_PROJECT_NAME}::
##)
##
### Register package in user's package registry
##export(PACKAGE la)
##
