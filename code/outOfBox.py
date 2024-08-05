import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import pickle 
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import spearmanr

def train_RF(features, target, model_file):
    try:
        # 尝试加载模型
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        # 如果模型文件不存在，则重新训练模型
        print("Model not found. Training a new model...")

        # 训练随机森林模型
        rf = RandomForestRegressor(random_state=0)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')

        # 训练模型
        grid_search.fit(features, target)

        # 输出最佳参数组合
        print(f"Best parameters found: {grid_search.best_params_}")

        # 使用最佳参数组合并启用 OOB 评分，重新定义随机森林回归模型
        best_rf = RandomForestRegressor(**grid_search.best_params_, oob_score=True, random_state=42)

        # 在整个训练集上训练最佳模型
        best_rf.fit(features, target)

        # 输出最佳 OOB 评分
        print(f"Best OOB Score: {best_rf.oob_score_}")

        # 保存训练好的模型
        with open(model_file,'wb') as f:
            pickle.dump(best_rf, f)
        print("Model trained and saved successfully.")

    return best_rf

def main():
    # 加载数据集
    dataSetName = '1β-glucosidase-train'
    path = '../data/数据集/' + dataSetName + '.csv'
    data = pd.read_csv(path, encoding = 'GBK')
    x_train = data[['βT', 'Rg', 'SASA', 'RMSD', 'DSI']]
    y_train = data['Tm']

    testpath = '../data/数据集/'+'1β-glucosidase-test.csv'
    test = pd.read_csv(testpath, encoding = 'GBK')
    x_test = test[['βT', 'Rg', 'SASA', 'RMSD', 'DSI']]
    y_test = test['Tm']

    # 归一化特征数据
    scaler = MinMaxScaler()

    # 划分数据集，模型训练的训练集，最终性能评估的测试集
    # 对于训练集，可再根据嵌套交叉验证二次划分
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)


    # 回归结果
    corr = []

    # 训练或加载模型

    model = train_RF(x_train,y_train,'../models/'+dataSetName+'-oufOfBox.pkl')
    # 回归预测
    predicted = model.predict(x_test)
    # 计算spearman相关系数
    corrlation,p_value = spearmanr(predicted,y_test)
    corr.append(corrlation)

    print(corr)


if __name__ == "__main__":
    main()