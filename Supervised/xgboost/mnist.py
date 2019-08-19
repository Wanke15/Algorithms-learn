import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import metrics

import xgboost as xgb

# The digits dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()

feature = digits.images.reshape(digits.images.shape[0], -1)
one_hot = OneHotEncoder()
# labels = one_hot.fit_transform(digits.target.reshape(-1, 1))
labels = digits.target.reshape(-1, 1)

train_x, valid_x, train_y, valid_y = train_test_split(feature, labels, test_size=0.333, random_state=0)
print(train_x.shape, train_y.shape)
print(valid_x.shape, valid_y.shape)

train = xgb.DMatrix(train_x, train_y)
valid = xgb.DMatrix(valid_x, valid_y)  # train函数下需要传入一个Dmatrix值，具体用法如代码所示

params = {
    'max_depth': 5,
    'learning_rate': 0.2,
    'n_estimators': 2,
    'min_child_weight': 4,
    'max_delta_step': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0,
    'reg_lambda': 0.4,
    'scale_pos_weight': 0.8,
    'silent': True,
    'objective': 'multi:softmax',
    'num_class': 10,
    'missing': None,
    'eval_metric': 'mlogloss',
    'seed': 1440,
    'gamma': 0
}  # 这里的params特指booster参数，注意这个eva_metric是评估函数

xlf = xgb.train(params, train, evals=[(valid, 'eval')],
                num_boost_round=2000, early_stopping_rounds=30, verbose_eval=True)
# 训练，注意验证集的写法， 还有early_stopping写法，这里指的是30轮迭代中效果未增长便停止训练
val_d = xgb.DMatrix(valid_x)
y_pred = xlf.predict(valid, ntree_limit=xlf.best_ntree_limit)
# xgboost没有直接使用效果最好的树作为模型的机制，这里采用最大树深限制的方法，目的是获取刚刚early_stopping效果最好的，实测性能可以

tr_acc_score = metrics.accuracy_score(train_y, xlf.predict(train, ntree_limit=xlf.best_ntree_limit))
print(tr_acc_score)

te_acc_score = metrics.accuracy_score(valid_y, y_pred)
print(te_acc_score)

# Gridsearch
parameters = {
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'n_estimators': [500, 1000, 2000, 3000, 5000],
    'min_child_weight': [0, 2, 5, 10, 20],
    'max_delta_step': [0, 0.2, 0.6, 1, 2],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

}

xlf = xgb.XGBClassifier(silent=True,
                        objective='multi:softmax',
                        nthread=-1,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)

gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

