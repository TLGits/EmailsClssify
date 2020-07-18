import time
import argparse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics

if __name__ == "__main__":

    # 可配置式运行文件
    parser = argparse.ArgumentParser(description='Choose the kernel and data_file')
    parser.add_argument('--method', '-m', help='method 选择核函数(linear or rbf or poly or sigmoid)', default = 'rbf')
    parser.add_argument('--file', '-f', help='file 选择采用数据集类型(stemming or lemmatization)', default='stemming')
    args = parser.parse_args()
    method = args.method
    file = args.file

    data = pd.read_table(file + '_preed_data.csv', header=0, encoding='utf-8', sep=",", index_col=0)
    data = data.as_matrix()[1:][1:]
    X, y = data[:, 0:-1], data[:, -1].astype(int)

    kf = KFold(n_splits=5, shuffle = True) # 五折训练，打乱数据顺序

    precisionList = []
    recallList = []
    f1List = []

    i = 1 # 表征五折训练的次数

    time_start = time.time()
    print(file + "_preed_data.csv 数据读取完成，开始五折SVM训练，method = "+ method)

    for train_index, val_index in kf.split(X):

        time_kfold_start = time.time()
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

        # 建立模型
        clf = svm.SVC(kernel=method, gamma = 'scale', cache_size = 800)

        # 模型拟合
        clf.fit(X_train, y_train)

        #模型预测
        y_pred = clf.predict(X_val)

        # 计算评价指标：precision, recall, f1_score
        precision = metrics.precision_score(y_val, y_pred)
        recall = metrics.recall_score(y_val, y_pred)
        f1 = metrics.f1_score(y_val, y_pred)
        precisionList.append(precision)
        recallList.append(recall)
        f1List.append(f1)

        time_kfold_end = time.time()
        print("KFold_" + str(i))
        print("time consumption:" + str(time_kfold_end - time_kfold_start) + "s")
        print("precision:" + str(precision))
        print("recall:" + str(recall))
        print("f1:" + str(f1))
        i += 1

    time_end = time.time()
    print("************五折训练全部完成************")
    print("total time consumption:" + str(time_end - time_start) + "s")
    print("average precision:" + str(np.mean(precisionList)))
    print("average recall:" + str(np.mean(recallList)))
    print("average f1:" + str(np.mean(f1List)))