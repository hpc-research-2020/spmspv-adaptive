#define LOG_NUM_BANKS 5 
#define NUM_BANKS 32 
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)
#define AVOID_CONFLICT(lid) ((lid) + ((lid) >> LOG_NUM_BANKS))

//src[i] <= d_csc_col[d_sparse_x_key[i]+1] - d_csc_col[d_sparse_x_key[i]]
template<int NB_LSIZE, int NB_LSIZE_LOG, int H_NB_LSIZE, 
        int NB_LSIZE_1, int NB_LSIZE_2,
        int NB_CTA_NUM, int NB_CTA_SIZE, int STEP_LOG, int H_NB_CTA_SIZE, 
        int H_NB_CTA_SIZE_1, int NB_CTA_SIZE_2, int NB_CTA_SIZE_1, 
        int NB_CTA_LOG, int NB_CTA_LOG_1>
__global__ void scanFuseKernel(
        int* d_csc_col,  int x_nnz, int* d_sparse_x_key,
        int* src, volatile int* sum, int* dst, int groupnum){
// __global__ void scanKernel(int* src,volatile int* sum, 
//                             int* dst,int groupnum) {
  //unsigned int lid = get_local_id(0); 
  unsigned int lid = threadIdx.x;
#if defined DYNAMIC_TASK
  unsigned int gid;
  //__local int gid_;
  __shared__ int gid_;
  if(lid == 0)
      gid_ = atomicAdd((__global__ int* )(sum + groupnum), 1);//sum[groupnum]
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
  gid = gid_;
#else
 //unsigned int gid = get_group_id(0);
 unsigned int gid = blockIdx.x;
#endif
//NB_LOCAL_SIZE
//NB_REG_SIZE
//NB_CTA_NUM and NB_CTA_SIZE
//NB_CTA_NUM, NB_CTA_SIZE: ok
//NB_LSIZE_2：
//TODO: 
  // __local int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
  // __local int column[2][NB_LSIZE_2],lpsum;
  __shared__ int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
  __shared__ int column[2][NB_LSIZE_2], lpsum;
  int re[NB_REG_SIZE];
  int kgp = lid >> NB_CTA_LOG;
  int kid = lid & NB_CTA_SIZE_1;
  int lid_= lid + 1;
  
  int src_id = gid * STEP_NUM * NB_LSIZE 
             + kgp * NB_CTA_SIZE * STEP_NUM + kid;
  int t1 = 0, t2 = 0, t3 = 0, t4 = 0, psum = 0, rsum = 0;
#if NB_REG_GRP   
   //registergroup.
  for (t1 = 0; t1 < NB_REG_GRP; t1++) {
    t3 = NB_CTA_SIZE * t1 + src_id;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      //src[i] <= d_csc_col[d_sparse_x_key[i]+1] - d_csc_col[d_sparse_x_key[i]]
      //lm[kgp][t2][kid] = src[t3 + t2 * STEP_NUM];
      lm[kgp][t2][kid] = d_csc_col[d_sparse_x_key[t3 + t2 * STEP_NUM] + 1] - d_csc_col[d_sparse_x_key[t3 + t2 * STEP_NUM]];
    }
    t3 = NB_CTA_SIZE * t1;
    re[t3] = lm[kgp][kid][0] + psum;
    for (t2 = 1; t2 < NB_CTA_SIZE; t2++) {
        re[t2 + t3] = re[t2 - 1 + t3] + lm[kgp][kid][t2];
    }
    psum = re[t3 + NB_CTA_SIZE_1];
    rsum = psum;
  }
#endif
  // localgroup.
  for (t1 = 0; t1 < NB_LOCAL_GRP; t1++) {
    t3 = NB_CTA_SIZE * (t1 + NB_REG_GRP) + src_id;
    t4 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      //lm[kgp][t2][t4 + kid] = src[t3 + t2 * STEP_NUM];
      lm[kgp][t2][t4 + kid] = d_csc_col[d_sparse_x_key[t3 + t2 * STEP_NUM] + 1] - d_csc_col[d_sparse_x_key[t3 + t2 * STEP_NUM]];
    }   
    for(t2 = 0; t2 < NB_CTA_SIZE; t2++)
        psum += lm[kgp][kid][t4 + t2];
  }
  column[0][lid_] = psum;
   
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
 
  for(t1 = 1, t2 = 1, t3 = 1; t1 <= H_NB_LSIZE; 
    t1 <<= 1, t2 <<= 1, t3 = t3^1) {
    column[t3][lid_] = lid >= t1 
        ? column[t3^1][lid_] + column[t3^1][lid_ - t2] 
        : column[t3^1][lid_];
    //barrier(CLK_LOCAL_MEM_FENCE);
    __syncthreads();
  }
  psum = 0; 
  t3 = t3^1;
  
  if (lid == 0) {
    if (gid == 0)
        sum[0] = column[t3][NB_LSIZE];
    else {
        while ((psum = sum[gid - 1]) == 0){}
        sum[gid] = column[t3][NB_LSIZE] + psum;
    }
    lpsum = psum;
    column[t3][0] = 0; 
  }
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
  psum = lpsum + column[t3][lid];

  lm[kgp][kid][0] = lm[kgp][kid][0] + psum + rsum;
  for (t1 = 1; t1 < NB_LOCAL_SIZE; t1++)
    lm[kgp][kid][t1] += lm[kgp][kid][t1-1];

  for (t1 = 0; t1 < NB_LOCAL_GRP; t1++) {
    t3 = NB_CTA_SIZE * (t1 + NB_REG_GRP) + src_id;
    t4 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      //dst[t3 + t2 * STEP_NUM] = lm[kgp][t2][t4 + kid];
      dst[t3 + t2 * STEP_NUM + 1] = lm[kgp][t2][t4 + kid];
    }   
  }

#if NB_REG_GRP
  for (t1 = 0; t1 < NB_REG_GRP; t1++) {
    t3 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      lm[kgp][kid][t2] = re[t2 + t3] + psum;
    }
    t3 = NB_CTA_SIZE * t1 + src_id;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      //dst[t3 + t2 * STEP_NUM] = lm[kgp][t2][kid];
      dst[t3 + t2 * STEP_NUM + 1] = lm[kgp][t2][kid];
    }
  }
#endif   
 if (gid * blockDim.x + lid == 0) {
  dst[0] = 0;
 } 
}

template<int NB_LSIZE, int NB_LSIZE_LOG, int H_NB_LSIZE, 
        int NB_LSIZE_1, int NB_LSIZE_2,
        int NB_CTA_NUM, int NB_CTA_SIZE, int STEP_LOG,
        int H_NB_CTA_SIZE, int H_NB_CTA_SIZE_1, int NB_CTA_SIZE_2, 
        int NB_CTA_SIZE_1, int NB_CTA_LOG, int NB_CTA_LOG_1>
__global__ void scanFuseTail( 
      int* d_csc_col,  int x_nnz, int* d_sparse_x_key,
      int* src, volatile int* sum, int* dst, 
      int elemnum, int pregroupnum, 
      int groupnum, int len) {
  //unsigned int lid = get_local_id(0);
  unsigned int lid = threadIdx.x;
#if defined DYNAMIC_TASK
  unsigned int gid;
  //__local int gid_;
  __shared__ int gid_;
  if(lid == 0)
    gid_ = atomicAdd((__global__ int*)(sum + groupnum + pregroupnum + 1), 1);
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
  gid = gid_;
#else
 //unsigned int gid = get_group_id(0);
  unsigned int gid = blockIdx.x;
#endif
  // __local int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
  // __local int column[2][NB_LSIZE_2],lpsum;
  __shared__ int lm[NB_CTA_NUM][NB_CTA_SIZE][TAIL_NB_LOCAL_SIZE];
  __shared__ int column[2][NB_LSIZE_2],lpsum;
  int re[TAIL_NB_REG_SIZE];
  int kgp = lid >> NB_CTA_LOG;
  int kid = lid & NB_CTA_SIZE_1;
  int lid_ = lid + 1;
  int src_id = gid * TAIL_STEP_NUM * NB_LSIZE 
             + kgp * NB_CTA_SIZE * TAIL_STEP_NUM 
             + kid + elemnum - len;
  int t1 = 0, t2 = 0, t3 = 0, t4 = 0, psum = 0, rsum = 0;
#if TAIL_NB_REG_GRP   
  for (t1 = 0; t1 < TAIL_NB_REG_GRP; t1++) {
    t3 = NB_CTA_SIZE * t1 + src_id;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      if (t3 + t2 * TAIL_STEP_NUM < elemnum)
        //lm[kgp][t2][kid] = src[t3 + t2 * TAIL_STEP_NUM];
        lm[kgp][t2][kid] = d_csc_col[d_sparse_x_key[t3 + t2 * TAIL_STEP_NUM] + 1] - d_csc_col[d_sparse_x_key[t3 + t2 * TAIL_STEP_NUM]];
      else
        lm[kgp][t2][kid] = 0;
    }
    t3 = NB_CTA_SIZE * t1;
    re[t3] = lm[kgp][kid][0] + psum;
    for (t2 = 1; t2 < NB_CTA_SIZE; t2++) {
      re[t2 + t3] = re[t2 - 1 + t3] + lm[kgp][kid][t2];
    }
    psum = re[t3 + NB_CTA_SIZE_1];
    rsum = psum;
  }
#endif
  for (t1 = 0; t1 < TAIL_NB_LOCAL_GRP; t1++) {
    t3 = NB_CTA_SIZE * (t1 + TAIL_NB_REG_GRP) + src_id;
    t4 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      if (t3 + t2 * TAIL_STEP_NUM < elemnum)
        //lm[kgp][t2][t4 + kid] = src[t3 + t2 * TAIL_STEP_NUM];
        lm[kgp][t2][t4 + kid] = d_csc_col[d_sparse_x_key[t3 + t2 * TAIL_STEP_NUM] + 1] - d_csc_col[d_sparse_x_key[t3 + t2 * TAIL_STEP_NUM]];//src[t3 + t2 * TAIL_STEP_NUM];
      else
        lm[kgp][t2][t4 + kid] = 0;
    }   
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++)
      psum += lm[kgp][kid][t4 + t2];
  }
  column[0][lid_] = psum;
   
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
  for (t1 = 1, t2 = 1, t3 = 1; t1 <= H_NB_LSIZE; 
      t1 <<= 1, t2 <<= 1, t3 = t3^1) {
    column[t3][lid_] = lid >= t1 
                     ? column[t3^1][lid_] + column[t3^1][lid_ - t2] 
                     : column[t3^1][lid_];
    //barrier(CLK_LOCAL_MEM_FENCE);
    __syncthreads();
  }
  psum = 0; 
  t3 = t3^1;
  if (lid == 0) {
    if(gid == 0){
      if (pregroupnum != 0)
        psum = sum[pregroupnum-1];
      else
        psum = 0;
      sum[pregroupnum+1] = psum + column[t3][NB_LSIZE];
    }
    else {
      while ((psum=sum[pregroupnum + gid]) == 0) {}
      sum[pregroupnum + 1 + gid] = column[t3][NB_LSIZE] + psum;
    }
    lpsum = psum;
    column[t3][0] = 0;
  }
  //barrier(CLK_LOCAL_MEM_FENCE);
  __syncthreads();
  psum = lpsum + column[t3][lid];
  lm[kgp][kid][0] = lm[kgp][kid][0] + psum + rsum;
  for (t1 = 1; t1 < TAIL_NB_LOCAL_SIZE; t1++)
    lm[kgp][kid][t1] += lm[kgp][kid][t1-1];
  for (t1 = 0; t1 < TAIL_NB_LOCAL_GRP; t1++) {
    t3 = NB_CTA_SIZE * (t1 + TAIL_NB_REG_GRP) + src_id;
    t4 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      if (t3 + t2 * TAIL_STEP_NUM < elemnum)
        //dst[t3 + t2 * TAIL_STEP_NUM] = lm[kgp][t2][t4 + kid];
        dst[t3 + t2 * TAIL_STEP_NUM + 1] = lm[kgp][t2][t4 + kid];
    }   
  }
#if TAIL_NB_REG_GRP
  for (t1 = 0; t1 < TAIL_NB_REG_GRP; t1++) {
    t3 = NB_CTA_SIZE * t1;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      lm[kgp][kid][t2] = re[t2 + t3] + psum;
    }
    t3 = NB_CTA_SIZE * t1 + src_id;
    for (t2 = 0; t2 < NB_CTA_SIZE; t2++) {
      if (t3 + t2 * TAIL_STEP_NUM < elemnum)
        //dst[t3 + t2 * TAIL_STEP_NUM] = lm[kgp][t2][kid];
        dst[t3 + t2 * TAIL_STEP_NUM + 1] = lm[kgp][t2][kid];
    }
  }
#endif    
  if (gid * blockDim.x + lid == 0)
    dst[0] = 0;
}
//#endif