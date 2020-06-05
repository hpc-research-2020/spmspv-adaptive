#include "class.hpp"

template<class T>
bool if_sorted_coo(MTX<T>* mtx)
{
    int nnz = mtx->nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mtx->row[i] > mtx->row[i+1]) || (mtx->row[i] == mtx->row[i+1] 
              && mtx->col[i] > mtx->col[i+1]))
            return false;
    }
    return true;
}

template<class T>
bool sort_coo(MTX<T>* mtx)
{

    int i = 0;
    int beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    int pivrow, pivcol;
    T pivdata;

    beg[0] = 0; 
    end[0] = mtx->nnz;
    while (i>=0) 
    {
        L = beg[i];
        //TODO: 
        if (end[i] - 1 > end[i])
            R = end[i];
        else
            R = end[i] - 1;
        if (L<R) 
        {
            int middle = (L+R)/2;
            pivrow=mtx->row[middle]; 
            pivcol=mtx->col[middle];
            pivdata=mtx->data[middle];
            mtx->row[middle] = mtx->row[L];
            mtx->col[middle] = mtx->col[L];
            mtx->data[middle] = mtx->data[L];
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            if (i==MAX_LEVELS-1) 
                return false;
            while (L<R) 
            {
                while (((mtx->row[R] > pivrow) || 
                            (mtx->row[R] == pivrow && mtx->col[R] > pivcol)) 
                        && L<R) 
                    R--; 
                if (L<R) 
                {
                    mtx->row[L] = mtx->row[R];
                    mtx->col[L] = mtx->col[R];
                    mtx->data[L] = mtx->data[R];
                    L++;
                }
                while (((mtx->row[L] < pivrow) || 
                            (mtx->row[L] == pivrow && mtx->col[L] < pivcol)) 
                        && L<R) 
                    L++; 
                if (L<R) 
                {
                    mtx->row[R] = mtx->row[L];
                    mtx->col[R] = mtx->col[L];
                    mtx->data[R] = mtx->data[L];
                    R--;
                }
            }
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            beg[i+1]=L+1; 
            end[i+1]=end[i]; 
            end[i++]=L; 
        }
        else 
        {
            i--; 
        }
    }
    return true;
}


template<class T>
bool if_sorted_col_coo(MTX<T>* mtx)
{
    int nnz = mtx->nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mtx->col[i] > mtx->col[i+1]) || (mtx->col[i] == mtx->col[i+1] 
              && mtx->row[i] > mtx->row[i+1]))
            return false;
    }
    return true;
}

template<class T>
bool sort_col_coo(MTX<T>* mtx)
{

    int i = 0;
    int beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    int pivrow, pivcol;
    T pivdata;

    beg[0]=0; 
    end[0]=mtx->nnz;
    while (i>=0) 
    {
        L=beg[i];
        if (end[i] - 1 > end[i])
            R = end[i];
        else
            R = end[i] - 1;
        if (L<R) 
        {
            int middle = (L+R)/2;
            pivrow=mtx->row[middle]; 
            pivcol=mtx->col[middle];
            pivdata=mtx->data[middle];
            mtx->row[middle] = mtx->row[L];
            mtx->col[middle] = mtx->col[L];
            mtx->data[middle] = mtx->data[L];
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            if (i==MAX_LEVELS-1) 
                return false;
            while (L<R) 
            {
                while (((mtx->col[R] > pivcol) || 
                            (mtx->col[R] == pivcol && mtx->row[R] > pivrow)) 
                        && L<R) 
                    R--; 
                if (L<R) 
                {
                    mtx->row[L] = mtx->row[R];
                    mtx->col[L] = mtx->col[R];
                    mtx->data[L] = mtx->data[R];
                    L++;
                }
                while (((mtx->col[L] < pivcol) || 
                            (mtx->col[L] == pivcol && mtx->row[L] < pivrow)) 
                        && L<R) 
                    L++; 
                if (L<R) 
                {
                    mtx->row[R] = mtx->row[L];
                    mtx->col[R] = mtx->col[L];
                    mtx->data[R] = mtx->data[L];
                    R--;
                }
            }
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            beg[i+1]=L+1; 
            end[i+1]=end[i]; 
            end[i++]=L; 
        }
        else 
        {
            i--; 
        }
    }
    return true;
}

template<class T>
void printSeperateMtx2File(int rows, int cols, int nnz, int* row_id, int* col_id, T* val)
{
    std::ofstream fout;
    fout.open("seperatemtx.out");
    //fout << "rows:" << mtx->rows << "  cols:" << mtx->cols << "  non zeros:" << mtx->nnz << std::endl;
     fout << rows << "  " << cols << "  " << nnz << std::endl;
    for(int i = 0; i < nnz; i++){
       fout << row_id[i] << "  " << col_id[i] << "  " << val[i] << std::endl;
    }
    fout.close();
}

template <typename T>
	struct Entry
	{
		unsigned r, c;
		T v;
		bool operator < (const Entry& other)
		{
			if (r != other.r) 
				return r < other.r;
			return c < other.c;
		}
	};

template<typename T>
bool compareAsc(const Entry<T> &value1,const Entry<T> &value2) {
    //return value1 < value2;
  if(value1.r != value2.r)
    return value1.r < value2.r;
  return value1.c < value2.c;
}

template<typename T>
bool compareAscColPri(const Entry<T> &value1,const Entry<T> &value2) {
    //return value1 < value2;
  if(value1.c != value2.c)
    return value1.c < value2.c;
  return value1.r < value2.r;
}

template<class T>
void ReadMTX(const char* filename,MTX<T> *mtx, bool isRowSorted)
{
    FILE* infile = fopen(filename, "r");
    if (infile == NULL) {
      printf("open file error\n");
      exit(1);
    }
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);

    bool ifreal = false;
    bool ispattern = false;
    bool isinterger = false;
    bool iscomplex = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    else if(strcmp(tmpstr, "pattern") == 0)
            ispattern = true;
         else if(strcmp(tmpstr, "integer") == 0)
                isinterger = true;
              else
                iscomplex = true;

    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;

    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
        fscanf(infile, "%s", tmpstr);
        if (tmpstr[0] != '%')
        {
            height = atoi(tmpstr);
            break;
        }
        fgets(tmpline, 1025, infile);
    }

    fscanf(infile, "%d %d", &width, &nnz);
    mtx->rows = height;
    mtx->cols = width;

    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    T* data = (T*)malloc(sizeof(T)*nnz);
    
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = 1.0;
        if(ifreal) {
          double dbldata = 0.0f;
          fscanf(infile, "%lf", &dbldata);
          data[i] = (T)dbldata;
        }
        else if(isinterger){
            int dbldata = 0;
            fscanf(infile, "%d", &dbldata);
            data[i] = dbldata;
        }
        else if(ispattern){
               data[i] = 1.0;
        }
        else if(iscomplex){//is complex.
            double data1 = 0.0f;
            double data2 = 0.0f;
            fscanf(infile, "%lf", &data1);
            fscanf(infile, "%lf", &data2);
            data[i] = 1.0;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    //debug.
    //printSeperateMtx2File(height, width, nnz, rows, cols, data);

    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mtx->nnz = newnnz;
        mtx->row = (int*)malloc(sizeof(int)*newnnz);
        mtx->col = (int*)malloc(sizeof(int)*newnnz);
        mtx->data = (T*)malloc(sizeof(T)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mtx->row[matid] = rows[i];
            mtx->col[matid] = cols[i];
            mtx->data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mtx->row[matid] = cols[i];
                mtx->col[matid] = rows[i];
                mtx->data[matid] = data[i];
                matid++;
            }
        }
        if(matid != newnnz){
            std::cout<<"Error: matid != newnnz!"<<std::endl;
        }
    }
    else
    {
        mtx->nnz = nnz;
        mtx->row = (int*)malloc(sizeof(int)*nnz);
        mtx->col = (int*)malloc(sizeof(int)*nnz);
        mtx->data = (T*)malloc(sizeof(T)*nnz);
        memcpy(mtx->row, rows, sizeof(int)*nnz);
        memcpy(mtx->col, cols, sizeof(int)*nnz);
        memcpy(mtx->data, data, sizeof(T)*nnz);
    }

    //printSeperateMtx2File(height, width, newnnz, mtx->row, mtx->col, mtx->data);
    fclose(infile);
    free(rows);
    free(cols);
    free(data);

    //sort 
    bool tmp = false;
    
    std::vector<Entry<T>> entries;
    entries.reserve(mtx->nnz);
    for (int i = 0; i < mtx->nnz; ++i) {
      // Entry<T> item;
      // item.r = mtx->row[i];
      // item.c = mtx->col[i];
      // item.v = mtx->data[i];
      // entries.push_back(item);
      entries.push_back(Entry<T>{ mtx->row[i], mtx->col[i], mtx->data[i] });
    }
      
    std::sort(std::begin(entries), std::end(entries), compareAsc<T>);

    for (int i = 0; i < mtx->nnz; ++i) {
      mtx->data[i] = entries[i].v;
      mtx->col[i] = entries[i].c;
      mtx->row[i] = entries[i].r;
    }

    tmp = if_sorted_coo<T>(mtx);
    if(tmp != true){
      std::cout<<"Error: not sorted!"<<std::endl;
    }

    return;    
}

template<class T>
void SortMTX(MTX<T> *mtx, bool isRowSorted)
{
  bool tmp = false;
  
  std::vector<Entry<T>> entries;
	entries.reserve(mtx->nnz);
	for (size_t i = 0; i < mtx->nnz; ++i)
		entries.push_back(Entry<T>{ mtx->row[i], mtx->col[i], mtx->data[i] });
  std::sort(std::begin(entries), std::end(entries), compareAsc<T>);

  for (size_t i = 0; i < mtx->nnz; ++i) {
		mtx->data[i] = entries[i].v;
		mtx->col[i] = entries[i].c;
    mtx->row[i] = entries[i].r;
	}

  tmp = if_sorted_coo<T>(mtx);
  if(tmp != true){
    std::cout<<"Error: not sorted!"<<std::endl;
  }
return;    
}

template<class T>
void fileToMtxCoo(const char* filename, MTX<T> *mtx, bool isRowSorted)
{
  ReadMTX<T>(filename, mtx, isRowSorted);
  SortMTX<T>(mtx, isRowSorted);
}

template<class T>
void printMtx(MTX<T> *mtx)
{
    std::cout<<"rows:"<<mtx->rows<<"  cols:"<<mtx->cols<<"  non zeros:"<<mtx->nnz<<std::endl;
    for(int i=0;i<mtx->nnz;i++){
       std::cout<<mtx->col[i]<<"  "<<mtx->row[i]<<"  "<<mtx->data[i]<<std::endl;
    }
}

template<class T>
void printMtx2File(MTX<T> *mtx)
{
    std::ofstream fout;
    fout.open("mtx.out");
   
    //fout << "rows:" << mtx->rows << "  cols:" << mtx->cols << "  non zeros:" << mtx->nnz << std::endl;
     fout << mtx->rows << "  " << mtx->cols << "  " << mtx->nnz << std::endl;
    for(int i=0; i<mtx->nnz; i++){
       fout << mtx->col[i] << "  " << mtx->row[i] << "  " << mtx->data[i] << std::endl;
    }
    fout.close();
}

 //add coo to hybrid format interface.
 //coo format to hybrid-csr-csc format
 template <class T>
 void COO2Hybrid(int row_priority, int nrows, int ncols, int nnz, 
                int* coo_row, int* coo_col, T* coo_val, int* ret_csr_nnz) {
   int* elems_per_row = (int*)malloc(nnz * sizeof(int));
   int* elems_per_col = (int*)malloc(nnz * sizeof(int));
    
   int csr_nnz = 0;
   int csc_nnz = 0;
   int new_csr_nnz = 0;
   int new_csc_nnz = 0;
   
   int err = SPMSPV_SUCCESS;
   
   memset(elems_per_row, 0, nnz*sizeof(int));
   memset(elems_per_col, 0, nnz*sizeof(int));
   for(int i=0; i<nnz; i++){
    int row_id = coo_row[i];
    int col_id = coo_col[i];
    elems_per_row[row_id]++;
    elems_per_col[col_id]++;
   }
    

#if 1
   if(row_priority){
     for(int i=0; i<nnz; i++){
       int row_id = coo_row[i];
       int col_id = coo_col[i];
       if(elems_per_row[row_id] >= elems_per_col[col_id]){
         //add to csr nnzs clusters.
         csr_nnz += 1;
         //swap i's nnz to csr_nnz's pos.
         elems_per_col[col_id]--;
       }else{
         //add to csc nnzs clusters.
         csc_nnz += 1;
         elems_per_row[row_id]--;
       } 
     }
   }else{
     for(int i=0; i<nnz; i++){
       int row_id = coo_row[i];
       int col_id = coo_col[i];
       if(elems_per_row[row_id] < elems_per_col[col_id]){
         //add to csc nnzs clusters.
         csc_nnz += 1;
         elems_per_col[row_id]--;
       }else{
         //add to csr nnzs clusters.
         csr_nnz += 1;
         elems_per_row[col_id]--;
       } 
     }
   
   }
   printf("csr_nnz = %d, csc_nnz = %d, sum = %d, csr_nnz_ratio=%lf.\n", csr_nnz, csc_nnz, csr_nnz+csc_nnz, csr_nnz/(1.0*nnz));
#endif
   //csr's: in front:
   //csc's: in tail
   std::vector<Entry<T>> entries_csr;
   std::vector<Entry<T>> entries_csc;
   entries_csr.reserve(csr_nnz);
   entries_csc.reserve(csc_nnz);

#if 0	
   entries.reserve(mtx->nnz);
	for (size_t i = 0; i < mtx->nnz; ++i)
		entries.push_back(Entry<T>{ mtx->row[i], mtx->col[i], mtx->data[i] });
  std::sort(std::begin(entries), std::end(entries), compareAsc<T>);
#endif


   memset(elems_per_row, 0, nnz*sizeof(int));
   memset(elems_per_col, 0, nnz*sizeof(int));
   for(int i=0; i<nnz; i++){
    int row_id = coo_row[i];
    int col_id = coo_col[i];
    elems_per_row[row_id]++;
    elems_per_col[col_id]++;
   }
   
   if(row_priority){
     for(int i=0; i<nnz; i++){
       int row_id = coo_row[i];
       int col_id = coo_col[i];
       if(elems_per_row[row_id] >= elems_per_col[col_id]){
         //add to csr nnzs clusters.
		     entries_csr.push_back(Entry<T>{ coo_row[i], coo_col[i], coo_val[i] });
         //csr_nnz += 1;
         //swap i's nnz to csr_nnz's pos.
         elems_per_col[col_id]--;
       }else{
         //add to csc nnzs clusters.
		     entries_csc.push_back(Entry<T>{ coo_row[i], coo_col[i], coo_val[i] });
         //csc_nnz += 1;
         elems_per_row[row_id]--;
       } 
     }
   }else{
     for(int i=0; i<nnz; i++){
       int row_id = coo_row[i];
       int col_id = coo_col[i];
       if(elems_per_row[row_id] < elems_per_col[col_id]){
         //add to csc nnzs clusters.
		     entries_csc.push_back(Entry<T>{ coo_row[i], coo_col[i], coo_val[i] });
         //csc_nnz += 1;
         elems_per_col[row_id]--;
       }else{
         //add to csr nnzs clusters.
		     entries_csr.push_back(Entry<T>{ coo_row[i], coo_col[i], coo_val[i] });
         //csr_nnz += 1;
         elems_per_row[col_id]--;
       } 
     }
   
   }


#if 0 
   int i = 0;
   int j = i + 1;
   int temp_i;
   TYPE_VT temp_val;
   bool break_out = false;
   int row_id_i, col_id_i, row_id_j, col_id_j;
   
   if(row_priority){
     while(i<j && i<nnz){
       row_id_i = coo_row[i];
       col_id_i = coo_col[i];
       if(elems_per_row[row_id_i] < elems_per_col[col_id_i]){
         //find nonzeros in col clusters
         elems_per_col[row_id_i]--;
         while(j<nnz){
           row_id_j = coo_row[j];
           col_id_j = coo_col[j];
           if(elems_per_row[row_id_j] >= elems_per_col[col_id_j]){
             //find nnzs in row clusters
             elems_per_col[col_id_j]--;
             break; 
           }else{
             elems_per_col[row_id_j]--;
             j++;
           }
         }

         if(j < nnz){
           //swap [i] and [j]
           temp_i = row_id_i;
           row_id_i = row_id_j;
           row_id_j = temp_i;

           temp_i = col_id_i;
           col_id_i = col_id_j;
           col_id_j = temp_i;

           temp_val = coo_val[i];
           coo_val[i] = coo_val[j];
           coo_val[j] = temp_val;

           i++;
           j++;
         }else{
           break_out = true;
           break;
         }
       }else{
         elems_per_col[col_id_i]--;
         i++;
         j++;
       }
     }
     //i now belongs to col clustes, but can not find j.
     if(break_out){
       //col
       new_csr_nnz = i;
     }else{
       new_csr_nnz = i;
     }
     new_csc_nnz = nnz - new_csr_nnz;
   }else{
     
     while(i<j && i<nnz){
       row_id_i = coo_row[i];
       col_id_i = coo_col[i];
       //i in row clusters
       if(elems_per_row[row_id_i] >= elems_per_col[col_id_i]){
         elems_per_col[col_id_i]--;
         //find j-th nonzeros in col clusters
         while(j<nnz){
           row_id_j = coo_row[j];
           col_id_j = coo_col[j];
           if(elems_per_row[row_id_j] < elems_per_col[col_id_j]){
             //find nnzs in row clusters
             elems_per_col[row_id_j]--;
             break; 
           }else{
             elems_per_col[col_id_j]--;
             j++;
           }
         }

         if(j < nnz){
           //swap [i] and [j]
           temp_i = row_id_i;
           row_id_i = row_id_j;
           row_id_j = temp_i;

           temp_i = col_id_i;
           col_id_i = col_id_j;
           col_id_j = temp_i;

           temp_val = coo_val[i];
           coo_val[i] = coo_val[j];
           coo_val[j] = temp_val;

           i++;
           j++;
         }else{
           break_out = true;
           break;
         }
       }else{
         elems_per_col[row_id_i]--;
         i++;
         j++;
       }
     }
     //i now belongs to col clustes, but can not find j.
     if(break_out){
       //col
       new_csc_nnz = i;
     }else{
       new_csc_nnz = i;
     }
     new_csr_nnz = nnz - new_csc_nnz;
   
   } 
   printf("new_csr_nnz = %d, new_csc_nnz = %d.\n", new_csr_nnz, new_csc_nnz);
   if(row_priority){
    //[0, i-1] is csr, [i, nnz-1] is csc: csr does not need to sort;
   }else{
    //[0, i-1] is csc, [i, nnz-1] is csr;
   }
#endif

#if 1 
   std::sort(std::begin(entries_csr), std::end(entries_csr), compareAsc<T>);
   std::sort(std::begin(entries_csc), std::end(entries_csc), compareAscColPri<T>);
   size_t i = 0; 
   for (i = 0; i < csr_nnz; ++i) {
		coo_val[i] = entries_csr[i].v;
		coo_col[i] = entries_csr[i].c;
    coo_row[i] = entries_csr[i].r;
	 }
   for (size_t j = 0; j < csc_nnz; ++j, ++i) {
		coo_val[i] = entries_csc[j].v;
		coo_col[i] = entries_csc[j].c;
    coo_row[i] = entries_csc[j].r;
   } 
#endif
   *ret_csr_nnz = csr_nnz;

   free(elems_per_row);
   free(elems_per_col);
   return ;
 }




//add csr io interface: ok
namespace {
	struct CSRIOHeader
	{
		static constexpr char Magic[9] = { 'H', 'o', 'l', 'a', 1, 'C', 'S', 'R', 1 };

		char magic[sizeof(Magic)];
		//uint64_t typesize;
		int num_rows, num_columns;
		int num_non_zeroes;

		CSRIOHeader() = default;
#if 0
		template<typename T>
		static uint64_t typeSize()
		{
			return sizeof(T);
		}
#endif
		//template<typename T>
		CSRIOHeader(int m, int n, int nnz)
		{
			for (size_t i = 0; i < sizeof(Magic); ++i)
				magic[i] = Magic[i];
			//typesize = typeSize<T>();

			num_rows = m;
			num_columns = n; 
			num_non_zeroes = nnz;
		}
#if 1 
		bool checkMagic() const
		{
			for (size_t i = 0; i < sizeof(Magic); ++i)
				if (magic[i] != Magic[i])
					return false;
			return true;
		}
#endif
	};
	constexpr char CSRIOHeader::Magic[];
}

int loadCSR_header(const char * file, int* m, int* n, int *nnz)
{
	std::ifstream fstream(file, std::fstream::binary);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");

	CSRIOHeader header;
	fstream.read(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
	if (!fstream.good())
		throw std::runtime_error("Could not read CSR header");
	if (!header.checkMagic())
		throw std::runtime_error("File does not appear to be a CSR Matrix");
#if 0
	if (header.typesize != CSRIOHeader::typeSize<T>())
		throw std::runtime_error("File does not contain a CSR matrix with matching type");
#endif
  
  *m = header.num_rows;
  *n = header.num_columns;
  *nnz = header.num_non_zeroes;

	return 0;
}


template<typename T>
int loadCSR(const char * file, int m, int n, int nnz, int* csr_row, int* csr_col, T* csr_val)
{
	std::ifstream fstream(file, std::fstream::binary);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");

	CSRIOHeader header;
	fstream.read(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
	if (!fstream.good())
		throw std::runtime_error("Could not read CSR header");
	if (!header.checkMagic())
		throw std::runtime_error("File does not appear to be a CSR Matrix");
#if 0
	if (header.typesize != CSRIOHeader::typeSize<T>())
		throw std::runtime_error("File does not contain a CSR matrix with matching type");
#endif
  
	//CSR<T> res;
	//res.alloc(header.num_rows, header.num_columns, header.num_non_zeroes);

	fstream.read(reinterpret_cast<char*>(&csr_val[0]), nnz * sizeof(T));
	fstream.read(reinterpret_cast<char*>(&csr_col[0]), nnz * sizeof(int));
	fstream.read(reinterpret_cast<char*>(&csr_row[0]), (m+1) * sizeof(int));

	if (!fstream.good())
		throw std::runtime_error("Could not read CSR matrix data");

	return 0;
}

template<class T>
void storeCSR(int m, int n, int nnz, int* csr_row, int* csr_col, T* csr_val, const char * file)
{
	std::ofstream fstream(file, std::fstream::binary);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");

	CSRIOHeader header(m, n, nnz);
	fstream.write(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));

	fstream.write(reinterpret_cast<char*>(&csr_val[0]), nnz * sizeof(T));
	fstream.write(reinterpret_cast<char*>(&csr_col[0]), nnz * sizeof(int));
	fstream.write(reinterpret_cast<char*>(&csr_row[0]), (m + 1) * sizeof(int));
}

template void COO2Hybrid(int row_priority, int nrows, int ncols, int nnz, 
            int* coo_row, int* coo_col, float* coo_val, int* ret_csr_nnz);

template int loadCSR(const char * file, int m, int n, int nnz, int* csr_row, int* csr_col, float* csr_val);
template int loadCSR(const char * file, int m, int n, int nnz, int* csr_row, int* csr_col, double* csr_val);
template void storeCSR(int m, int n, int nnz, int* csr_row, int* csr_col, float* csr_val, const char * file);
template void storeCSR(int m, int n, int nnz, int* csr_row, int* csr_col, double* csr_val, const char * file);
