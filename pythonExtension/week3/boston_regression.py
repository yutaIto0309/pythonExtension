import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot

np.random.seed(42)

# 訓練用データ
data_train = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/pythonExtension/week3/boston_train.csv', encoding='utf-8')
# テスト用データ
data_test = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/pythonExtension/week3/boston_test.csv', encoding='utf-8')

# print(data_train.loc[:5])

# 正則化なし学習
transformer = StandardScaler()
X_train = transformer.fit_transform(data_train.iloc[:, :-1].values)
y_train = data_train['PRICE'].values
regr = LinearRegression()
regr.fit(X_train, y_train)
print(f'weifhts:{regr.coef_}')
print(f'norm:{np.linalg.norm(regr.coef_)}')

# 正則化なし学習 性能評価
X_test = transformer.fit_transform(data_test.iloc[:, :-1].values)
y_test = data_test['PRICE'].values
y_pred = regr.predict(X_test)
print(f'RMSE:{mean_squared_error(y_test, y_pred)}')
print(f'R2:{r2_score(y_test, y_pred)}')

