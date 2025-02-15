import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import uniform

# CSVファイルの読み込み
csv_file_path = "4_wine_quality.csv"
df = pd.read_csv(csv_file_path)

# 品質を二値分類に変換（品質が5以下を0、6以上を1とする）
df['quality_binary'] = (df['quality'] >= 6).astype(int)

# 特徴量とターゲット変数の準備
xi_column_list = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
y_column_list = ["quality_binary"]
xi = df[xi_column_list]
y = df[y_column_list]

# データの分割
xi_train, xi_test, y_train, y_test = train_test_split(
    xi, y, test_size=0.2, random_state=123
)

# ランダムサーチによるハイパーパラメータの最適化
param_dist = {
    'C': uniform(loc=0.1, scale=9.9),  # 正則化パラメータの範囲（0.1から10）
    'solver': ['lbfgs', 'liblinear']  # ソルバーの選択
}

logistic_regression = LogisticRegression(max_iter=1000)
random_search = RandomizedSearchCV(
    logistic_regression, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=123
)
random_search.fit(xi_train, y_train.values.ravel())

# 最適なハイパーパラメータ
best_params = random_search.best_params_
print(f'最適なハイパーパラメータ: {best_params}')

# 最適なモデルの評価
best_model = random_search.best_estimator_
y_pred = best_model.predict(xi_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

# 各変数の最尤推定量（回帰係数）を取得し、データフレームに格納
coefficients = pd.DataFrame({
    'Feature': xi_column_list,
    'Coefficient': best_model.coef_[0]
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("\n各変数の最尤推定量（回帰係数）:")
print(coefficients)

# 回帰係数を棒グラフで表示
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Logistic Regression Coefficients')
plt.gca().invert_yaxis()  # 特徴量を上から下に並べる
plt.show()

# 結果の表示
results = pd.DataFrame(random_search.cv_results_)
plt.figure(figsize=(12, 6))

# Accuracyのプロット
plt.subplot(1, 2, 1)
for solver in param_dist['solver']:
    subset = results[results['param_solver'] == solver]
    plt.scatter(subset['param_C'], subset['mean_test_score'], label=f'Solver: {solver}')
plt.xscale('log')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Mean Test Accuracy')
plt.title('Mean Test Accuracy vs. C')
plt.legend()

plt.tight_layout()
plt.show()
