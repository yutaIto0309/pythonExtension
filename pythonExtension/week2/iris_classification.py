import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# アヤメのデータ読み込み
iris = load_iris()

clf = LogisticRegression(solver='liblinear', C=1e+2, multi_class='auto')
clf.fit(iris.data, iris.target)




