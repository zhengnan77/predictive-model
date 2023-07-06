import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut

df = pd.read_csv('input.csv', index_col=0, sep=',')
print(df.shape[1])
X = df.values[:, :-1]
df_norm = (df-df.min())/(df.max()-df.min())
df_stand = (df-df.mean())/df.std()
X_norm, y_norm = df_norm.values[:, :-1], df_norm.values[:, -1]
X_stand, y_stand = df_stand.values[:, :-1], df_stand.values[:, -1]


rf = RandomForestRegressor()
rf_cv_scores = cross_val_score(rf, X_norm, y_norm, scoring='r2', cv=5)
print('rf_cv_scores=', rf_cv_scores, ',', 'mean score=', np.mean(rf_cv_scores), ',', 'std.dev.=', np.std(rf_cv_scores))

parameters = {'max_features': np.arange(0.1, 1.1, 0.1)}
rf_tuned = GridSearchCV(rf, parameters)
rf_tuned_cv_scores = cross_val_score(rf_tuned, X_norm, y_norm, scoring='r2', cv=5)
print('rf_tuned_cv_scores=', rf_tuned_cv_scores, ',', 'mean score=', np.mean(rf_tuned_cv_scores), ',', 'std.dev.=', np.std(rf_tuned_cv_scores))
print('y_predicted vs. [y_measured]')
rf_tuned.fit(X, y)
import pickle
pickle.dump(rf_tuned, open('trained_model.sav', 'wb'))
import pandas as pd
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
df_new = pd.read_csv('input.csv', index_col=0, sep=',')
X_new = df_new.values
y_new_predicted = loaded_model.predict(X_new)
df_new['predicted_activity'] = y_new_predicted
df_new.to_csv('predicted_values.csv')
print(rf_tuned.predict(X_new))
