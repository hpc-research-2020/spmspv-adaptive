#ifndef CLASS_H
#define CLASS_H

//#define TEXTURE_WIDTH 512
#define MAX_LEVELS 1000
//#define TIMES 2 
//#define SH_MEM_SIZE 48*1024
//#define MAX_WORKGROUP_SIZE 512
//#define ESTIMATE

template <class dataType>
struct MTX{
    int rows;
    int cols;
    int nnz;
    int *row;
    int *col;
    dataType* data;
};

template<typename T>
struct Node{
    Node(int index, T value) : index(index), value(value) {}
    int index;
    T value;
};

template<typename T>
struct SvmData {
  unsigned int numVects;
  unsigned int dimVects;
  unsigned int numNonZeros;

  typedef std::vector<std::vector<Node <T > > > rowFeature;
  std::vector<int> label_;
  rowFeature samples_;
};



#if 0
template<class dataType,class dimType,class bitType>
struct BCCOO{
    int rows;
    int cols;
    int nnz;
    
    int block_number;
    int block_width;
    int block_height;
   
    int max_block_per_row;
    int slice_rows;
    int slices;

    bitType *bit;
    dimType *col;
    dataType *data;
};

struct CLBCCOO{
    int rows;
    int cols;
    int nnz;
   
    int block_number;
    int block_width;
    int block_height;
   
    int max_block_per_row;
    int slice_rows;
    int slices;
   
    cl_mem res_entry;
    cl_mem col_delta;
    cl_mem bit;
    cl_mem col;
    cl_mem data;
    cl_mem data1;
    cl_mem data2;
    cl_mem data3;
    cl_mem para_scan;
    cl_mem inter;
};
#endif

#endif //CLASS_H

