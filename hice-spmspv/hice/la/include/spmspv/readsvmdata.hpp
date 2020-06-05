/*
Author: 
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <float.h>
#include <typeinfo>
#include <limits>
#include <algorithm>
#include <vector>

#include <omp.h>

#include "spmspv/class.hpp"
//#define MAXLEN 6000000

// template<typename T>
// struct Node;

// template<typename T>
// struct SvmData;
// template<typename T>
// struct Node{
//     Node(int index, T value) : index(index), value(value) {}
//     int index;
//     T value;
// };

// template<typename T>
// struct SvmData {
//   unsigned int numVects;
//   unsigned int dimVects;
//   unsigned int numNonZeros;

//   typedef std::vector<std::vector<Node <T > > > rowFeature;
//   std::vector<int> label_;
//   rowFeature samples_;
// };

inline char *findlastline(char *ptr, char *begin) {
       while (ptr != begin && *ptr != '\n') --ptr;
       return ptr;
}

template<typename T>
int readSVMToCoo(char* file_name, SvmData<T>* svmdata) {
  std::cout << "(multi-thread) loading svm dataset from file " 
            << file_name << std::endl;
  typedef std::vector<std::vector<Node <T > > > rowFeature;
  
  svmdata->label_.clear();
  svmdata->samples_.clear();
  svmdata->numVects = 0;
  svmdata->dimVects = 0;
  svmdata->numNonZeros = 0;
  // total_count_ = 0;
  // n_features_ = 0;
    
  std::ifstream ifs(file_name, std::ifstream::binary);
  if(!ifs.is_open()) 
  	std::cout << "file " << file_name << " not found" << std::endl;
  	//std::cout<< "file " << file_name << " not found"<<std::endl;//TODO

  int buffer_size = 16 << 20; //16MB
  char *buffer = (char *)malloc(buffer_size);
  const int nthread = omp_get_max_threads();

  while (ifs) {
      char *head = buffer;
      ifs.read(buffer, buffer_size);
      size_t size = ifs.gcount();
      std::vector<std::vector<int>> label_thread(nthread);
      std::vector<rowFeature> samples_thread(nthread);

      std::vector<int> local_feature(nthread, 0);
#pragma omp parallel num_threads(nthread)
      {
          //get working area of this thread
          int tid = omp_get_thread_num();
          size_t nstep = (size + nthread - 1) / nthread;
          size_t sbegin = min(tid * nstep, size);
          size_t send = min((tid + 1) * nstep, size);
          //ok:懂啦，之前被面试过这个问题。多线程读取文件。
          char *pbegin = findlastline(head + sbegin, head);
          char *pend = findlastline(head + send, head);

          //move stream start position to the end of last line
          if (tid == nthread - 1) ifs.seekg(pend - head - send + 1, std::ios_base::cur);
          //read instances line by line
          char *lbegin = pbegin;
          char *lend = lbegin;
          while (lend != pend) {
              //get one line
              lend = lbegin + 1;
              while (lend != pend && *lend != '\n') {
                  ++lend;
              }
              std::string line(lbegin, lend);////Copies the sequence of characters in the range [first,last), in the same order.
              std::stringstream ss(line);

              //read label of an instance
              label_thread[tid].emplace_back();//
              ss >> label_thread[tid].back();//

              //read features of an instance
              samples_thread[tid].emplace_back();//
              std::string tuple;
              while (ss >> tuple) {
                  int i;
                  float v;
                  if (sscanf(tuple.c_str(), "%d:%f", &i, &v) != 2) {
                  	std::cout << "read error, using [index]:[value] format" 
                              << std::endl;
                  	exit(0);
                  }
                  samples_thread[tid].back().emplace_back(i, v);//
                  svmdata->numNonZeros++;
                  if (i > local_feature[tid]) local_feature[tid] = i;
              };

              //read next instance
              lbegin = lend;
          }
      }
      for (int i = 0; i < nthread; i++) {
          if (local_feature[i] > svmdata->dimVects)
              svmdata->dimVects = local_feature[i];
          svmdata->numVects += samples_thread[i].size();
      }
      for (int i = 0; i < nthread; i++) {
          svmdata->label_.insert(svmdata->label_.end(), label_thread[i].begin(), label_thread[i].end());
          svmdata->samples_.insert(svmdata->samples_.end(), samples_thread[i].begin(), samples_thread[i].end());
      }
  }
  free(buffer);
  //std::cout<<"#instances = "<<this->GetNumVects()<<", #features = "<<this->GetDimVects()<<std::endl;
  std::cout << "#instances = " << svmdata->numVects << ", #features = " 
            << svmdata->dimVects << std::endl;

//read right??
//#define CHECK_INDEX_VALUE
#ifdef CHECK_INDEX_VALUE
  std::cout << "check index-value result\n" << std::endl;
  std::ofstream fout;
  fout.open("check_index_value.out");
  
  int i, j;
  int index;
  for (i = 0; i < svmdata->samples_.size(); i++) {
    fout << svmdata->label_[i] << " ";
    for (j = 0; j < svmdata->samples_[i].size(); j++) {
        index = svmdata->samples_[i][j].index;
        fout << index << ":" << svmdata->samples_[i][j].value << " ";
    }
    fout << std::endl;
  }
  fout.close();
#endif
  
  return 0;
}

template<typename T>
int freeSVMToCoo(SvmData<T>* svmdata){
  #if 1 
  //释放vector的内存
  std::vector<int>().swap(svmdata->label_);
  for (int i = 0; i < svmdata->samples_.size(); i++){
    std::vector<Node<T>>().swap(svmdata->samples_[i]); 
  }
  std::vector<std::vector<Node<T>>>().swap(svmdata->samples_); 
  printf("label_.capacity()=%d\n", svmdata->label_.capacity());
  printf("sample_.capacity()=%d\n", svmdata->samples_.capacity());
  
#endif
return 0;
}


template<typename T>
int ConvertSVMDataToCSR(SvmData<T> svmdata, 
                        int* csr_row, int* csr_col, T* csr_val, 
                        int numVects, int dimVects, int numNonZeros) {
  int i, j;
  unsigned int offset = 0;
  csr_row[0] = 0;
  
  for (i = 0; i < svmdata.samples_.size(); i++){
    for (j = 0; j < svmdata.samples_[i].size(); j++){
        csr_val[offset] = svmdata.samples_[i][j].value;
        csr_col[offset] = svmdata.samples_[i][j].index - 1;//
        offset++;
    }
    csr_row[i+1] = offset;
  }

  for (int ii = i + 1; ii < numVects+1; ii++) 
    csr_row[ii] = offset; //fill the padded area


  printf("NNZ: %d\n%% sparsity: %.3lf\n Avg. NNZ per row: %.3lf\n", 
      numNonZeros, 100.0 * numNonZeros / (numVects * dimVects), 
      numNonZeros / (double)numVects);
  return 0;
} //SvmData::ConvertDataToCSR

void getNumofRows(char* svm_file_name, int* numRows){
  std::cout << "get num of rows (nIter)." << std::endl;
  
  std::ifstream ifs(svm_file_name, std::ifstream::binary);
  if (!ifs.is_open()) 
    std::cout << "file " << svm_file_name << " not found" << std::endl;;
  int index;
  int length = 0;  
  while (ifs >> index) {    
    length++;
  }
  *numRows = length;
}

void readNNZXFromFile(const char* file_name, int* len) {
  std::cout << "start readNNZXFromFile" << std::endl;

  std::ifstream ifs(file_name, std::ifstream::binary);
  if (!ifs.is_open()) 
    std::cout << "file " << file_name << " not found" << std::endl;
  
  int index;
  int length = 0;  
  while (ifs >> index) {    
    length++;
  }
  *len = length;
}

void readSparseXInxFromFile(const char* svm_file_name, int len, int* ind) {
  std::cout << "start readSparseXInxFromFile" << std::endl;

  std::ifstream ifs(svm_file_name, std::ifstream::binary);
  if (!ifs.is_open()) 
    std::cout << "file " << svm_file_name << " not found" << std::endl;
  
  int index;
  int length = 0;  
  while (ifs >> index) {    
    //std::cout << "Read from file: " << index << std::endl;  
    ind[length++] = index;
  }
  //std::ofstream fout;
  //fout.open("key.out");
  
  //for (int i = 0; i < length; i++) {
  //  fout << ind[i] << std::endl;
  //}
  //fout.close();
}

//是否已经存在？
bool isExist(int val, int len, int* arr) {
  for (int i = 0; i < len; i++) {
    if (val == arr[i])
      return true;
  }
  return false;
}
void removeRepetition(int oldLen, int* old_ind, int* newLen, int* new_ind) {
  int insertPos = 1;
  for (int i = 1; i < oldLen; i++) {
    if (!isExist(old_ind[i], insertPos, new_ind)) {
      new_ind[insertPos++] = old_ind[i];
    }
  }
  *newLen = insertPos;
  // std::ofstream fout;
  // fout.open("new-key.out");
  
  // for (int i = 0; i < *newLen; i++) {
  //   fout << new_ind[i] << std::endl;
  // }
  // fout.close();
}

template<typename iT, typename vT>
void extractSparseXfromMat(int row_id, 
                          int* csr_row, int* csr_col, vT* csr_val, 
                          int m, int n, int mat_nnz, 
                          iT* x_sparse_key, vT* x_sparse_val) {
  //std::cout << "start read sparse x from file" << std::endl;                            
  int ii = 0;
  for (int i = csr_row[row_id]; i < csr_row[row_id+1]; i++) {
    x_sparse_key[ii] = csr_col[i];
    x_sparse_val[ii] = csr_val[i];
    ii++;
  }

  // std::ofstream fout;
  // fout.open("key-value.out");
  
  // for (int i = 0; i < ii; i++) {
  //   fout << x_sparse_key[i] << " : " << x_sparse_val[i] << std::endl;
  // }
  // fout.close();
}


template <typename iT, typename vT>
void extractSparseXfromFile(std::string file_name, int l_num, int x_nnz,
                            iT *x_sparse_key, vT *x_sparse_val, vT* x_dense) {
    std::ifstream ifs(file_name, std::ifstream::binary);
    if (!ifs.is_open())
        std::cout << "file " << file_name << " not found" << std::endl;
    int count = 0;
    int key = 0;
    while (!ifs.eof()) {
        std::string strline;
        // ifs.getline(stringline, MAXLEN);
        std::getline(ifs, strline);
        if (count == l_num) {
            std::stringstream ss(strline);
            std::string tuple;
            //parse current l_num.
            int i = 0;
            while (ss >> tuple) {
                if (sscanf(tuple.c_str(), "%d", &key) != 1) {
                    std::cout << "read error" << std::endl;
                    exit(0);
                }
                x_sparse_key[i] = key;
                x_sparse_val[i] = 1.0;
                x_dense[key] = 1.0;
                i++;
            }
        }
        count++;
    }
    ifs.close();

    //check.
    // std::ofstream fout;
    // fout.open("key.out");
    // for (int i = 0; i < x_nnz; i++)
    // {
    //     fout << x_sparse_key[i] << std::endl;
    // }
    // fout.close();
}
