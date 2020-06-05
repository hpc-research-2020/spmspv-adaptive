import os
import os.path
import sys

import numpy as np
import time

from sklearn.model_selection import GridSearchCV
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
##bin_len, x_sparisy.
#idx = [11,14]
#idx = [0,1,2,3,4,5,6,7,8,9,10,11,14]
idx = [0,1,2,3,4,5,6,7,8,9,10,11,14]
#idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

# features： 7 + 9 + 5 + 1.
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,maxCol,minCol,avgCol,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3".split(",")
#[0,13]
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,stdRow,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3,xnnz/n,nnz_s/nnz".split(",")
#[0,14]:add GM1/GM3
#feature_names = "m,n,nnz,maxRow,minRow,avgRow,stdRow,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3,GM1/GM3,xnnz/n,nnz_s/nnz".split(",")
feature_names = "m,n,nnz,maxRow,minRow,avgRow,range,stdRow,edge_equlity,gini_coefficiency,x_nnz,bin_len,max,min,xnnz/n, bin_len/nnz,xnnz_range,GM1,GM2,GM3,GM1/GM2,GM2/GM3,GM1/GM3".split(",")
feature_names = np.array(feature_names)[idx]
# print(feature_names)
def draw_tree(model, name):
  dot_data = StringIO()
  _tree.export_graphviz(model, out_file = dot_data)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  graph.write_pdf(name + ".pdf")

# def get_code(tree, feature_names):
#   left      = tree.tree_.children_left
#   right     = tree.tree_.children_right
#   threshold = tree.tree_.threshold
#   features  = [feature_names[i] for i in tree.tree_.feature]
#   value = tree.tree_.value

#   def recurse(left, right, threshold, features, node):
#     if (threshold[node] != -2):
#       print("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
#       if left[node] != -1:
#         recurse (left, right, threshold, features,left[node])
#       print("} else {")
#       if right[node] != -1:
#         recurse (left, right, threshold, features,right[node])
#       print("}")
#     else:
#       print("return " + str(value[node]))

#   recurse(left, right, threshold, features, 0)

def get_code(tree, feature_names):
  left      = tree.tree_.children_left
  right     = tree.tree_.children_right
  threshold = tree.tree_.threshold
  features  = [feature_names[i] for i in tree.tree_.feature]
  value = tree.tree_.value

  def recurse(left, right, threshold, features, node):
    if (threshold[node] != -2):
      print("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
      if left[node] != -1:
        recurse (left, right, threshold, features,left[node])
      print("} else {")
      if right[node] != -1:
        recurse (left, right, threshold, features,right[node])
      print("}")
    else:
      print("return " + str(np.array(value[node][0]).argsort()[-1]) + ";")

  recurse(left, right, threshold, features, 0)

def print_info(estimator):
  # print(estimator.get_params())
  print(estimator.feature_importances_)
  # print(estimator.max_features_)
  print(estimator.n_features_)
  print(estimator.classes_)
  print(estimator.decision_path)
  print(estimator.tree_)
  n_nodes = estimator.tree_.node_count
  children_left = estimator.tree_.children_left
  children_right = estimator.tree_.children_right
  feature = estimator.tree_.feature
  threshold = estimator.tree_.threshold
  value = estimator.tree_.value
  print(value)
  # The tree structure can be traversed to compute various properties such
  # as the depth of each node and whether or not it is a leaf.
  node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, -1)]  # seed is the root node id and its parent depth
  while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
      stack.append((children_left[node_id], parent_depth + 1))
      stack.append((children_right[node_id], parent_depth + 1))
    else:
      is_leaves[node_id] = True

  print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
  for i in range(n_nodes):
    if is_leaves[i]:
      print("%snode=%s leaf node. class%s" % (node_depth[i] * "\t", i, np.array(value[i][0]).argsort()[-1]))
    else:
      print("%snode=%s test node: go to node %s if `%s` <= %s else to "
      # print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
          "node %s."
          % (node_depth[i] * "\t",
           i,
           children_left[i],
           feature_names[feature[i]],
           threshold[i],
           children_right[i],
           ))
  print()

def save_model(model, target_file="t.model"):
  joblib.dump(model, target_file)

def load_model(model_file):
  return joblib.load(model_file)

def train_schedule_dtree():
  X, y = load_data("raw_full-model-3.dat")
  #X, y = load_data("predict.dat")
  X = X[:, idx]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

  #clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
  #clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
  clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight="balanced")

  #parameters = {'criterion':('gini', 'entropy'), 'max_depth':[1, 10]}
  #dt = DecisionTreeClassifier()
  #clf = GridSearchCV(dt, parameters)

  clf = clf.fit(X_train, y_train)
  result = clf.predict(X_test)
  print("y_test:", y_test)
  print("X_test predict result:", result)
  print()
  score = clf.score(X_test, y_test)
  #print_info(clf)
  #get_code(clf, feature_names)
  # draw_tree(clf, "learn")
  # tree_to_code(clf, feature_names)
  print("score:{}".format(score))
  save_model(clf, "schedule_dtree-model-3.m")

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

#def test_schedule(verbose=False):
def test_schedule(verbose=True):
  # vec_desc = map(lambda x: x.strip(), "m,n,nnz,maxRow,minRow,avgRow,maxCol,minCol,avgCol,x_nnz,bin_len,GM1,GM2,GM3,GM1/GM2,GM2/GM3".split(","))
  # vid = dict()
  # for i, desc in enumerate(vec_desc): vid[desc] = i
  clf = load_model("schedule_dtree-model-3.m")
  files = list_dir("test-model-3")
  #files = list_dir("data")
  all_score = 0
  average_score = 0
  num_files = 0
  loss_ratio = 0
  loss_ratio_real = 0

  all_sort_col = 0
  all_no_sort_col = 0
  all_predict_col = 0
  all_best_col = 0

  lm_all_time = 0

  for f in files:
    print(f)
    num_files = num_files+1
    X, y = load_data(f)
    X_test = X[:,idx]
    time_start = time.time()
    result = clf.predict(X_test)
    time_end = time.time()
    lm_all_time += time_end - time_start

    predict_time = 0.0
    best_time = 0.0

    no_sort_col = 0.0
    sort_col = 0.0

    for i in range(len(result)):
      line_data  = X[i]
      # print(len(line_data))
      #line_data[]: csc-spmspv,spmspv+s2a,spmv+s2d,spmspv,spmv
      #line_data[]: "naive-col,lb-col,naive-rspmspv,naive-rspmspv+s2a,lb-rspmspv,lb-rspmspv+s2a,naive-spmv,naive-spmv+s2d,lb-spmv,lb-spmv+s2d"
      c = [line_data[-2], line_data[-1]]

      indexs = c.index(min(c))
      best_time += c[indexs]

      predict_time += c[result[i]]

      sort_col += c[0]
      no_sort_col += c[1]

      if(verbose):
        if(y[i] !=result[i]):
          print("wrong predict: i: {}, real_label:{}, predict_label:{}".format(i, y[i], result[i]))
    print("best_time:{}, predict_time:{}, speedup:{}".format(best_time, predict_time, predict_time/best_time))
    score = clf.score(X_test, y)
    all_score += score
    all_sort_col += sort_col
    all_no_sort_col += no_sort_col
    all_predict_col += predict_time
    all_best_col += best_time
    print("{} score:{}".format(os.path.basename(f), score))
    print("\n")
    print("sort_col time:{}".format(sort_col))
    print("no-sort_col time:{}".format(no_sort_col))
    print("predict_col time:{}".format(predict_time))
    print("best_col time:{}".format(best_time))
    print("\n")
    loss_ratio += (predict_time - best_time)/best_time
  average_score = all_score/num_files
  print("average scores:{}".format(average_score))
  print("average loss_ratio:{}".format(loss_ratio/num_files))
  print("all_sort_col time:{}".format(all_sort_col))
  print("all_no-sort_col time:{}".format(all_no_sort_col))
  print("all_predict_col time:{}".format(all_predict_col))
  print("all_best_col time:{}".format(all_best_col))
  print("Time:{}".format(lm_all_time))

def main():
  #train_schedule_dtree()
  test_schedule()

if __name__ == '__main__':
  main()
