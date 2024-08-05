import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 创建示例数据
# 加载数据�?
dataSetName = '1��-glucosidase'
path = '../data/数据�?' + dataSetName + '.csv'
data = pd.read_csv(path, encoding = 'GBK')
features = data[['βT', 'Rg', 'SASA', 'RMSD', 'DSI','hydrophobicity','charge']]
target = data['Tm']

# 归一化特征数�?
scaler = MinMaxScaler()
normalFeatures = scaler.fit_transform(features)

# 分割数据�?
X_train, X_test, y_train, y_test = train_test_split(normalFeatures, target, test_size=0.2, random_state=52)

# 训练随机森林模型
rf = RandomForestRegressor(random_state=52)
rf.fit(X_train, y_train)

# 计算特征重要性（基于Gini重要性）
feature_importances_gini = rf.feature_importances_
importance_df_gini = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances_gini
}).sort_values(by='Importance', ascending=False)
print("Feature importances (Gini):")
print(feature_importances_gini)

# importance_df_gini.sort_values(by='Importance', ascending=False)
# 绘制特征重要性图（基于Gini重要性）
plt.figure(figsize=(10, 5))
plt.barh(importance_df_gini['Feature'], importance_df_gini['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance (Gini)')
plt.gca().invert_yaxis()
plt.show()


