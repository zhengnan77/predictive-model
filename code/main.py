import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import pickle 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
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
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

         # 外部交叉验证
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=32)
        outer_scores = []
        best_estimators = []

        for train_idx, test_idx in outer_cv.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

            # 内部交叉验证和网格搜索
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=32)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=inner_cv, scoring='r2')
            grid_search.fit(X_train, y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_
            best_estimators.append(best_model)

            # 在外部验证集上评估最佳模型
            test_y_pred = best_model.predict(X_test)
            corrlation,p_value = spearmanr(test_y_pred,y_test)
            outer_scores.append(corrlation)

        # 选择外部交叉验证得分最高的模型作为最终模型
        best_index = np.argmax(outer_scores)
        model = best_estimators[best_index]

        # print("RF nested cross-validation corr: ",outer_scores)
        print("Mean RF nested cross-validation corr: ",np.mean(outer_scores))
        # print("Standard RF nested cross-validation corr: ",np.std(outer_scores))
        # 保存训练好的模型
        with open(model_file,'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")

    return model


def train_SVM(features, target, model_file):
    try:
        # 尝试加载模型
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        # 如果模型文件不存在，则重新训练模型
        print("Model not found. Training a new model...")

        # 定义支持向量机模型和参数网格
        svm = SVR()
        param_grid_svm = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.2, 0.3],
            'kernel': ['linear', 'rbf', 'poly']
        }

        # 外部交叉验证
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=32)
        outer_scores = []
        best_estimators = []

        for train_idx, test_idx in outer_cv.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

            # 内部交叉验证和网格搜索
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=32)
            grid_search = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=inner_cv, scoring='r2')
            grid_search.fit(X_train, y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_
            best_estimators.append(best_model)

            # 在外部验证集上评估最佳模型
            test_y_pred = best_model.predict(X_test)
            corrlation,p_value = spearmanr(test_y_pred,y_test)
            outer_scores.append(corrlation)

        # 选择外部交叉验证得分最高的模型作为最终模型
        best_index = np.argmax(outer_scores)
        model = best_estimators[best_index]

        # print("SVM nested cross-validation corr: ",outer_scores)
        print("Mean SVM nested cross-validation corr: ",np.mean(outer_scores))
        # print("Standard SVM nested cross-validation corr: ",np.std(outer_scores))

        # 保存训练好的模型
        with open(model_file,'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")

    return model


def train_KNN(features, target, model_file):
    try:
        # 尝试加载模型
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        # 如果模型文件不存在，则重新训练模型
        print("Model not found. Training a new model...")

        # 定义K近邻模型和参数网格
        knn = KNeighborsRegressor()
        param_grid_knn = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform']
        }

        # 外部交叉验证
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=32)
        outer_scores = []
        best_estimators = []

        for train_idx, test_idx in outer_cv.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

            # 内部交叉验证和网格搜索
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=32)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=inner_cv, scoring='r2')
            grid_search.fit(X_train, y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_
            best_estimators.append(best_model)

            # 在外部验证集上评估最佳模型
            test_y_pred = best_model.predict(X_test)
            corrlation,p_value = spearmanr(test_y_pred,y_test)
            outer_scores.append(corrlation)

        # 选择外部交叉验证得分最高的模型作为最终模型
        best_index = np.argmax(outer_scores)
        model = best_estimators[best_index]


        # print("KNN nested cross-validation corr: ",outer_scores)
        print("Mean KNN nested cross-validation corr: ",np.mean(outer_scores))
        # print("Standard KNN nested cross-validation corr: ",np.std(outer_scores))
        # 保存训练好的模型
        with open(model_file,'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")

    return model


def train_Linear(features, target, model_file):
    try:
        # 尝试加载模型
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        # 如果模型文件不存在，则重新训练模型
        print("Model not found. Training a new model...")

           # 定义线性回归模型
        linear_reg = LinearRegression()
        param_grid_lr = {
            'fit_intercept': [True, False],
        }

        # 外部交叉验证
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=32)
        outer_scores = []
        best_estimators = []

        for train_idx, test_idx in outer_cv.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

            # 内部交叉验证和网格搜索
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=32)
            grid_search = GridSearchCV(estimator=linear_reg, param_grid=param_grid_lr, cv=inner_cv, scoring='r2')
            grid_search.fit(X_train, y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_
            best_estimators.append(best_model)

            # 在外部验证集上评估最佳模型
            test_y_pred = best_model.predict(X_test)
            corrlation,p_value = spearmanr(test_y_pred,y_test)
            outer_scores.append(corrlation)
        # 选择外部交叉验证得分最高的模型作为最终模型
        best_index = np.argmax(outer_scores)
        model = best_estimators[best_index]

        # print("Liner nested cross-validation corr: ",outer_scores)
        print("Mean Liner nested cross-validation corr: ",np.mean(outer_scores))
        # print("Standard Liner nested cross-validation corr: ",np.std(outer_scores))
        # 保存训练好的模型
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")

    return model

def main():
    # 加载数据集
    dataSetName = '1β-glucosidase'
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

    # x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=32)
    # print(x_test)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    # x_train = x_train.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # x_test = x_test.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)
    # 模型及对应的调用函数
    models = [
        ('RF', train_RF),
        ('SVM', train_SVM),
        ('KNN', train_KNN),
        ('Linear_Regression', train_Linear)
    ]

    # 回归结果
    corr = []

    # 训练或加载模型
    for model_name, train_func in models:
        model = train_func(x_train,y_train,'../models/'+dataSetName+'-'+model_name+'.pkl')
        # 回归预测
        predicted = model.predict(x_test)
        # 将 numpy 数组转换为 pandas DataFrame
        df = pd.DataFrame(predicted)

        # 使用 pandas.DataFrame.to_csv 将 DataFrame 保存到 CSV 文件
        df.to_csv('../result/'+dataSetName+'-'+model_name+'.csv', index=False, header=False)
        print(type(predicted))
        # 计算spearman相关系数
        corrlation,p_value = spearmanr(predicted,y_test)
        corr.append(corrlation)

    print(corr)


if __name__ == "__main__":
    main()