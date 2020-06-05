import sys
import os
import os.path

def list_dir(path, suffix ='.info'):
  path = os.path.abspath(path)
  assert os.path.exists(path)
  dirs = os.listdir(path)
  result = []
  for d in dirs:
    reald = os.path.join(path, d)
    l = os.listdir(reald)
    for line in l:
      if line.endswith(suffix):
        result.append(os.path.join(reald, line))
  return result

#TODO:
def read_matrix_meta_info(fname="feature_all.info"):
  matrix_features = dict()
  with open(fname, "r") as fd:
    lines = fd.readlines()
    for line in lines:
      sp_data = line.strip().split()
      #print(sp_data)
      assert len(sp_data) == 11
      name = sp_data[0].strip().split('.')
      matrix_features[name[0].strip()] = list(map(float, sp_data[1:11]))
  return matrix_features


global TEST
global BFS
ferr = open("generate_label-2.log", "a+")
def parser_info_data(fname):
  def get_label(naive, lb):
    c = [naive, lb]
    return c.index(min(c))
  def get_label_6(para_1, para_2, para_3, para_4, para_5, para_6):
    c = [para_1, para_2, para_3, para_4, para_5, para_6]
    index_min_c = c.index(min(c))
    if index_min_c == 0 or index_min_c == 1:
      return 0
    if index_min_c == 2 or index_min_c == 3:
      return 1
    if index_min_c == 4 or index_min_c == 5:
      return 2
  matix_meta = read_matrix_meta_info("new-feature.info")
  #print(matix_meta)

  with open(fname) as fd, open("{}.dat".format(fname), "w+") as fout:
    basename = os.path.basename(fname)
    mat_name = basename[:-len("_{}_perf.info".format(("bfs" if BFS else "pr") if TEST else "feature"))]
    #print(mat_name)
    if not mat_name in matix_meta.keys():
      ferr.write("Error:{} not found meta info in new-feature.info\n".format(basename))
      return
    lines = fd.readlines()
    #print(mat_name)
    assert len(lines) > 3
    # matrix_desc = lines[0].strip().split()
    # matrix_data = lines[1].strip().split()
    # if len(matrix_data) != len(matrix_desc):
    # 	ferr.write("{} matrix_desc len != matrix_data len".format(mat_name))
    # assert  matix_meta[mat_name][0] == float(matrix_data[1])
    # assert  matix_meta[mat_name][1] == float(matrix_data[2])
    # assert  matix_meta[mat_name][2] ==  float(matrix_data[3])
    vec_desc = map(lambda x: x.strip(), lines[2].strip().split(","))
    vid = dict()
    for i, desc in enumerate(vec_desc): vid[desc] = i
    #print(vid)
    used_vec_desc = "x_nnz,bin_len,max,min,xnnz/n,bin_len/nnz,xnnz_range,GM1,GM2,GM3,GM1/GM2,GM2/GM3,GM1/GM3".split(",")
    #used_label_desc = "naive-col,lb-col,naive-rspmspv,naive-rspmspv+s2a,lb-rspmspv,lb-rspmspv+s2a,naive-spmv,naive-spmv+s2d,lb-spmv,lb-spmv+s2d".split(",")
    used_label_desc = "naive-col,lb-col,naive-spmv,lb-spmv".split(",")
    #print(used_vec_desc)
    #print(used_label_desc)
    vec_idx = list(map(lambda x: vid[x.strip()], used_vec_desc))
    label_idx = list(map(lambda x: vid[x.strip()], used_label_desc))
    #print(vec_idx)  #0,3,4,5,6,7,8

    # conj the matrix information into raw
    for line in lines[3:]:
      #print(line)
      line_data = list(map(float, line.strip().split()))
      new_data = [mat_name] + matix_meta[mat_name]
      #print(new_data)
      for i in vec_idx:
        new_data.append(line_data[i])
      #add GM1/GM3.
      #new_data.append(line_data[4]/line_data[6])

      #print(mat_name)
      #print(matix_meta[mat_name][0])
      #print(matix_meta[mat_name][2])
      #new_data.append(line_data[vec_idx[0]]/matix_meta[mat_name][1])#xnnz/n
      #new_data.append(line_data[vec_idx[3]]/matix_meta[mat_name][2])#bin_len/nnz
      for i in label_idx:
        new_data.append(line_data[i])
      #print(line_data[label_idx[0]], line_data[label_idx[1]], line_data[label_idx[2]])
      #new_data.append(get_label(line_data[label_idx[0]], line_data[label_idx[1]]))
      new_data.append(get_label(line_data[label_idx[2]], line_data[label_idx[3]]))
      #print(new_data)
      fout.write("{}\n".format("\t".join(list(map(str, new_data)))))

def conj_matrix_info_into_raw(dirname="data", _suffix=".info"):
  test_files = list_dir(dirname, suffix=_suffix)
  for f in test_files:
    print(f)
    parser_info_data(f)
  print(test_files)

def merge_raw_data(fname="raw_full.dat", _suffix=".info.dat"):
  dat_files = list_dir("test-model-2" if TEST else "data-model-2", suffix=_suffix)
  with open(fname, "w") as fout:
    for fr in dat_files:
      for txt in open(fr, 'r'):
        fout.write(txt)


if __name__ == '__main__':
  global TEST
  if len(sys.argv) > 1 and sys.argv[1] == "--test":
    TEST = True
  else:
    TEST = False
  if TEST:
    ##TODO: bfs or pr
    #BFS = False
    BFS = True
    print("--test\n")
    conj_matrix_info_into_raw("test-model-2", _suffix=".info")
    merge_raw_data(fname="predict-model-2.dat",  _suffix=".info.dat")
  else:
    print("--train\n")
    conj_matrix_info_into_raw("data-model-2", _suffix=".info")
    merge_raw_data("raw_full-model-2.dat", ".info.dat")
