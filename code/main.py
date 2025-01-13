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
      
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
                     
            test_y_pred = best_model.predict(X_test)
            corrlation,p_value = spearmanr(test_y_pred,y_test)
            outer_scores.append(corrlation)

        best_index = np.argmax(outer_scores)
        model = best_estimators[best_index]

        # print("RF nested cross-validation corr: ",outer_scores)
        print("Mean RF nested cross-validation corr: ",np.mean(outer_scores))
        # print("Standard RF nested cross-validation corr: ",np.std(outer_scores))
             with open(model_file,'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")

    return model


def train_SVM(features, target, model_file):
    try:
       
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
      
        print("Model not found. Training a new model...")
      
def main():
 
    dataSetName = '1β-glucosidase'
    path = '../data/数据集/' + dataSetName + '.csv'
    data = pd.read_csv(path, encoding = 'GBK')
    x_train = data[['βT', 'Rg', 'SASA', 'RMSD', 'DSI']]
    y_train = data['Tm']

    testpath = '../data/数据集/'+'**.csv'
    test = pd.read_csv(testpath, encoding = 'GBK')
    x_test = test[['βT', 'Rg', 'SASA', 'RMSD', 'DSI']]
    y_test = test['Tm']


    scaler = MinMaxScaler()


    # x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=32)
    # print(x_test)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    # x_train = x_train.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # x_test = x_test.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)
  
    models = [
        ('RF', train_RF),
         ]

  
    corr = []

    for model_name, train_func in models:
        model = train_func(x_train,y_train,'../models/'+dataSetName+'-'+model_name+'.pkl')
       
        predicted = model.predict(x_test)
      
        df = pd.DataFrame(predicted)

      
        df.to_csv('../result/'+dataSetName+'-'+model_name+'.csv', index=False, header=False)
        print(type(predicted))
       
        corrlation,p_value = spearmanr(predicted,y_test)
        corr.append(corrlation)

    print(corr)


if __name__ == "__main__":
    main()
