from typing import *
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
from NBTfidf import NBTfidfVectorizer
import csv
from math import sqrt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch

model = None
data = pd.read_csv('svm_han_corpus.csv')


HAN = 'out/whole_model_han.pt'

han = torch.load(HAN)


clf = SVC()
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
report = sklearn.metrics.classification_report(y_true, y_pred)
weighted_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
rsme = sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
# return clf.best_estimator_, report, weighted_f1, macro_f1 # TODO go
# between these two for svm and linear
if model == 'LinearSVC' or model == 'LogisticRegression':
    avg_weights = np.mean(clf.coef_, axis=0)
    with open('weights.csv', 'w', newline='') as weight_csv:
        weight_csv = csv.writer(csv_file)
        weight_csv.writerow(['feature', 'weight'])
        weights_zipped = zip(feature_names, avg_weights)
        for name, weight in weights_zipped:
            weight_csv.writerow([name, weight])

if model != "svm":
    print(clf)
else:
     print(clf.best_estimator_)
metric = [report, weighted_f1, macro_f1, micro_f1, rsme]
for m in metric:
    print(m)