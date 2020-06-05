#include "mv.h"
//#include "cblas.h"

void mv(const int m, const int n, 
        const double alpha, const double *a, const int lda,
        const double *x, const int incx, const double beta,
        double *y, const int incy){
/*
  CBLAS_LAYOUT Layout;
  CBLAS_TRANSPOSE transa;
  Layout = CblasColMajor;
  transa = CblasNoTrans;
  cblas_dgemv(Layout, transa, m, n, alpha, 
              a, lda, x, incx, beta, y, incy );
*/
 }
