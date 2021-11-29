import numpy as np
import math
import pickle
import pandas as pd 
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor

file = open('covid_dataset.pkl', 'rb')
checkpoint = pickle.load(file)
file.close()

def fix(X):
    return np.nan_to_num(np.array(X, dtype=np.float64))

def norm(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)

def apply_norm_fix(X, should_norm):
    X = fix(X)
    if should_norm:
        X = norm(X)
    return X

X_train  = apply_norm_fix(checkpoint["X_train"], False)
y_train = apply_norm_fix(checkpoint["y_train_log_pos_cases"], False)
X_val = apply_norm_fix(checkpoint["X_val"], False)
y_val = apply_norm_fix(checkpoint["y_val_log_pos_cases"], False)
big_X = apply_norm_fix(list(X_train) + list(X_val), False)
big_y = apply_norm_fix(list(y_train) + list(y_val), False)
X_test = apply_norm_fix(checkpoint["X_test"], False)

epochs = 1000
best_valid = math.inf
best_model = None
params = {'n_estimators': np.linspace(10, 1000, 100, dtype=np.int32), 'max_depth': np.linspace(10, 100, 10, dtype=np.int32)}

algorithm = ETR(n_estimators=10000, n_jobs=5)
x = X_train
y = y_train
model = algorithm.fit(x, y)
preds = model.predict(X_val)
mse = mean_squared_error(y_val, preds)
print(mse)

test_pred = model.predict(X_test)

pd.DataFrame(test_pred).to_csv("predictions.csv", header=["cases"], index_label="id")