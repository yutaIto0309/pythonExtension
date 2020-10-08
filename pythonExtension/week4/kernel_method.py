import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

# データ生成
X = 5 * np.random.rand(200, 1)
y = np.sin(3 * X).ravel() + np.random.normal(0, 0.3, 200)

# データの図示
# plt.figure(figsize=(12,8))
# plt.scatter(X.flatten(), y, s=10)
# X_tmp = np.linspace(0, 5, 1000)
# plt.plot(X_tmp, np.sin(3 * X_tmp), '--g')
# plt.show()

# ガウシアンカーネルのバンド幅、正則化係数をCVで最適化する
params = {
    'alpha' : np.logspace(-3, 1, 5),
    'gamma' : np.logspace(-2, 2, 5) 
}
regr = GridSearchCV(KernelRidge(kernel='rbf'), cv=5, param_grid=params)
regr.fit(X, y)
pd.DataFrame(regr.cv_results_)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']].sort_values(by=['rank_test_score'], ascending=True)

# バンド幅を変えた時の振る舞い
X_plot = np.linspace(0, 5, 10000)
plt.figure(figsize=(12,8))
plt.scatter(X.flatten(), y, s=5, c='gray')
for gamma in params["gamma"]:
    regr = KernelRidge(kernel='rbf', alpha=0.1, gamma=gamma)
    regr.fit(X, y)
    plt.plot(X_plot, regr.predict(X_plot[:,np.newaxis]), label=f'gamma={gamma}')
plt.title('alpha=0.1')
plt.legend()
# plt.show()

# より大規模なデータをRFFを用いて学習
X = 5 *np.random.rand(5000, 1)
y = np.sin(3 * X).ravel() + np.random.normal(0, 0.3, 5000)
X_test = 5 *np.random.rand(5000, 1)
y_test = np.sin(3 * X_test).ravel() + np.random.normal(0, 0.3, 5000)

rbf_feature = RBFSampler(gamma=1, n_components=100, random_state=42)
X_features = rbf_feature.fit_transform(X)
regr_rff = Ridge(alpha=1.0)
regr_rff.fit(X_features, y)

# カーネルとRFFの比較
# カーネルのインスタンス
regr_full = KernelRidge(kernel='rbf', alpha=0.1, gamma=1.0)
regr_full.fit(X,y)

plt.figure(figsize=(12,8))
plt.scatter(X.flatten(), y, s=5, c='gray', alpha=0.2)
X_plot_features = rbf_feature.transform(X_plot[:,np.newaxis])
plt.plot(X_plot, regr_rff.predict(X_plot_features), label='RFF (num_features=100)')
plt.plot(X_plot, regr_full.predict(X_plot[:, np.newaxis]), label='full kernel')
plt.legend()
#plt.show()

num = 40
num_list = []
for i in range(4):
    num_list.append((num - 4)/(num))
    num -= 1
print(np.prod(num_list))
