# Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity

Despite numerous efforts for optimizing the performance of Sparse Matrix and Vector Multiplication (SpMV) on modern hardware architectures, few works are done to its sparse counterpart, Sparse Matrix and Sparse Vector Multiplication (SpMSpV), not to mention dealing with input vectors of varied sparsity. We propose an adaptive SpMV/SpMSpV framework, which can automatically select the appropriate SpMV/SpMSpV kernel on GPUs for any sparse matrix and vector at the runtime. Based on systematic analysis on key factors such as computing pattern, workload distribution and write-back strategy, eight candidate SpMV/SpMSpV kernels are encapsulated into the framework to achieve high performance in a seamless manner. A machine learning based kernel selector is designed to choose the kernel and adapt with the varieties of both the input and hardware. For more detailed information, please refer to our paper, which will be relesed latter.

# Dependency

  * nvcc
  * cmake
  * cub
  * ModernGPU



# Compile

```c++
  cd hice-spmspv/hice/la/;
  mkdir build;
  cd build;
  cmake ..;
  make -j4
```

# Run

```python

  #The execution scripts are in the following path: 
  cd hice-spmspv/hice/la/script

  #Generate training data set:
  run generate_mat_feature.sh
  run generate_model_data.sh

  #Train the models
  cd model
  ## generate the label
  python 1-generate_matrix_class.py
  ## train the decision tree model:
  python 1-test_dtree.py
  ## train svm/gbdt/randomforest models
  python 1-test-svm.py
  python 1-test-randomforest.py
  python 1-test-gbdt.py

  # run adaptive spmspv
  sh run_bfs_predict.sh

  #
```


# License

All of the source codes of this repo are released under MIT license.
