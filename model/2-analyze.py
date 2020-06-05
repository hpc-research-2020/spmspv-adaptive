import os
import os.path
import sys

import numpy as np

from sklearn.externals.six import StringIO
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as _tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def load_data(filename="raw_full.dat"):
  with open(filename, encoding='UTF-8') as fd:
    lines = fd.readlines()
    X = list(map(lambda xx: list(map(float, xx.strip().split()[1:-1])), lines))
    y = list(map(lambda x: int(x.strip().split()[-1]), lines))
    return np.array(X), np.array(y)

#idx = [0,1,2,5,8,9,10,11,12,13,14,15]
###use all features.
##idx = [i for i in range(17)]
###use all usefull features
#idx = [0,1,2,6,8,9,10,11,12,13,14]
###remove m,n,nnz
#idx = [6,8,9,10,11,12,13,14]
###remove m,n,nnz,GM3
#idx = [6,8,9,10,12,13,14]
###remove bin_len: no use.
#idx = [0,1,2,6,9,10,11,12,13,14]
###remove bin_len and GM1:no use.
#idx = [0,1,2,6,10,11,12,13,14]
###remove bin_len and GM1/GM2: no use and improve the perf!!! wuwuwu.
##idx = [0,1,2,6,9,10,11,12]
###remove stdRow
#idx = [0,1,2,8,9,10,11,12,13,14]
###remove GM1, GM2, GM3
#idx = [0,1,2,6,8,12,13,14]
###remove GM1/GM2, GM1/GM3
#idx = [0,1,2,6,8,9,10,11]

###bin_len, GM1, GM2, GM3
#idx = [0,1,2,3,4,5,6,7,12,13]
###replace GM1/GM2,GM1/GM3 with xnnz/n and nnz_s/nnz
#idx = [0,1,2,3,4,5,6,7,8,9,10,11,14,15]
#idx = [0,1,2,3,4,5,7,8,9,10,11,12,13]


#idx = [i for i in range(23)]
#idx = [0,1,2,14]
#idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


idx = [9]
# features： 7 + 9 + 5 + 1.
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,maxCol,minCol,avgCol,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3".split(",")
#[0,13]
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,stdRow,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3,xnnz/n,nnz_s/nnz".split(",")
#[0,14]:add GM1/GM3
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,stdRow,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3,GM1/GM3,xnnz/n,nnz_s/nnz".split(",")
feature_names = "m,n,nnz,maxRow,minRow,avgRow,range,stdRow,edge_equlity,gini_coefficiency,x_nnz,bin_len,max,min,xnnz/n, bin_len/nnz,xnnz_range,GM1,GM2,GM3,GM1/GM2,GM2/GM3,GM1/GM3".split(",")
feature_names = np.array(feature_names)[idx]

# print(feature_names)

def load_model(model_file):
  return joblib.load(model_file)

def train_schedule_dtree():
  X, y = load_data("raw_full-model-2.dat")
  X = X[:, idx]
  num = 0
  num_0 = 0
  num_1 = 0
  c0 = [0, 0]
  c1 = [0, 0]
  c2 = [0, 0]
  c3 = [0, 0]
  c4 = [0, 0]
  c5 = [0, 0]
  c6 = [0, 0]
  c7 = [0, 0]
  c8 = [0, 0]
  c9 = [0, 0]
  for i in range(len(X)):
  #for i in range(30):
    num = num + 1
    line_data  = X[i]
    gini = line_data[0]
    x_sparsity = gini
    print(line_data[0], y[i])

    if y[i] == 0:
      num_0 += 1
      if x_sparsity>0 and x_sparsity<0.01:
        c0[0] = c0[0] + 1
      elif x_sparsity>=0.01 and x_sparsity<0.02:
        c1[0] = c1[0] + 1
      elif x_sparsity>=0.02 and x_sparsity<0.04:
        c2[0] = c2[0] + 1
      elif x_sparsity>=0.04 and x_sparsity<0.06:
        c3[0] = c3[0] + 1
      elif x_sparsity>=0.06 and x_sparsity<0.08:
        c4[0] = c4[0] + 1
      elif x_sparsity>=0.08 and x_sparsity<0.1:
        c5[0] = c5[0] + 1
      elif x_sparsity>=0.1 and x_sparsity<0.5:
        c6[0] = c6[0] + 1
      elif x_sparsity>=0.5 and x_sparsity<1:
        c7[0] = c7[0] + 1
    elif y[i] == 1:
      num_1 += 1
      if x_sparsity>0 and x_sparsity<0.01:
        c0[1] = c0[1] + 1
      elif x_sparsity>=0.01 and x_sparsity<0.02:
        c1[1] = c1[1] + 1
      elif x_sparsity>=0.02 and x_sparsity<0.04:
        c2[1] = c2[1] + 1
      elif x_sparsity>=0.04 and x_sparsity<0.06:
        c3[1] = c3[1] + 1
      elif x_sparsity>=0.06 and x_sparsity<0.08:
        c4[1] = c4[1] + 1
      elif x_sparsity>=0.08 and x_sparsity<0.1:
        c5[1] = c5[1] + 1
      elif x_sparsity>=0.1 and x_sparsity<0.5:
        c6[1] = c6[1] + 1
      elif x_sparsity>=0.5 and x_sparsity<1:
        c7[1] = c7[1] + 1

  r0 = c0[0] + c0[1] + 0.0
  if r0 == 0:
    r0 = 1
  c0[0] = c0[0]/r0
  c0[1] = c0[1]/r0

  r1 = c1[0] + c1[1] + 0.0
  if r1 == 0:
    r1 = 1
  c1[0] = c1[0]/r1
  c1[1] = c1[1]/r1

  r2 = c2[0] + c2[1] + 0.0
  if r2 == 0:
    r2 = 1
  c2[0] = c2[0]/r2
  c2[1] = c2[1]/r2

  r3 = c3[0] + c3[1] + 0.0
  if r3 == 0:
    r3 = 1
  c3[0] = c3[0]/r3
  c3[1] = c3[1]/r3

  r4 = c4[0] + c4[1] + 0.0
  if r4 == 0:
    r4 = 1
  c4[0] = c4[0]/r4
  c4[1] = c4[1]/r4

  r5 = c5[0] + c5[1] + 0.0
  if r5 == 0:
    r5 = 1
  c5[0] = c5[0]/r5
  c5[1] = c5[1]/r5

  r6 = c6[0] + c6[1] + 0.0
  if r6 == 0:
    r6 = 1
  c6[0] = c6[0]/r6
  c6[1] = c6[1]/r6

  r7 = c7[0] + c7[1] + 0.0
  if r7 == 0:
    r7 = 1
  c7[0] = c7[0]/r7
  c7[1] = c7[1]/r7

  r8 = c8[0] + c8[1] + 0.0
  if r8 == 0:
    r8 = 1
  c8[0] = c8[0]/r8
  c8[1] = c8[1]/r8


  print(c0[0], c0[1])
  print(c1[0], c1[1])
  print(c2[0], c2[1])
  print(c3[0], c3[1])
  print(c4[0], c4[1])
  print(c5[0], c5[1])
  print(c6[0], c6[1])
  print(c7[0], c7[1])
  print(num_0, num_1)

def list_dir(path, suffix ='.info.dat'):
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

def main():
  train_schedule_dtree()

if __name__ == '__main__':
  main()
