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
  	std::cout << "file " << file_name << " not found" << std::cout;
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
              label_thread[tid].emplace_back();//开辟了一个空间，但是没有赋值。
              ss >> label_thread[tid].back();//赋值。

              //read features of an instance
              samples_thread[tid].emplace_back();//这里有点绕。开辟存储一行的feature的vector数组。
              std::string tuple;
              while (ss >> tuple) {
                  int i;
                  float v;
                  if (sscanf(tuple.c_str(), "%d:%f", &i, &v) != 2) {
                  	std::cout << "read error, using [index]:[value] format" 
                              << std::endl;
                  	exit(0);
                  }
                  samples_thread[tid].back().emplace_back(i, v);//每一行的features的插入。
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
}

template<typename T>
int freeSVMToCoo(SvmData<T>* svmdata){
  #if 1 
  //释放vector的内存
  std::vector<int>().swap(svmdata->label_);
  for(int i=0; i<svmdata->samples_.size(); i++){
    std::vector<Node<T>>().swap(svmdata->samples_[i]); 
  }
  std::vector<std::vector<Node<T>>>().swap(svmdata->samples_); 
  printf("label_.capacity()=%d\n", svmdata->label_.capacity());
  printf("sample_.capacity()=%d\n", svmdata->samples_.capacity());
#endif
}


template<typename T>
int ConvertSVMDataToCSR(SvmData<T> svmdata, 
                        int* csr_row, int* csr_col, T* csr_val, 
                        int numVects, int dimVects, int numNonZeros) {
  int i, j;
  unsigned int offset = 0;
  csr_row[0] = 0;
  
  for(i = 0; i < svmdata.samples_.size(); i++){
    for(j = 0; j < svmdata.samples_[i].size(); j++){
        csr_val[offset] = svmdata.samples_[i][j].value;
        csr_col[offset] = svmdata.samples_[i][j].index - 1;//这里因为把libsvm中index为1的存储在数组的0号位置
        offset++;
    }
    csr_row[i+1] = offset;
  }

  for(int ii=i+1; ii < numVects+1; ii++) 
    csr_row[ii] = offset; //fill the padded area


  printf("NNZ: %d\n%% sparsity: %.3lf\n Avg. NNZ per row: %.3lf\n", 
      numNonZeros, 100.0 * numNonZeros / (numVects * dimVects), 
      numNonZeros / (double)numVects);
  return 0;
} //SvmData::ConvertDataToCSR
