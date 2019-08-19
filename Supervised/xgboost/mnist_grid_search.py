from sklearn import datasets
from sklearn import metrics

import xgboost as xgb

# The digits dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


digits = datasets.load_digits()

feature = digits.images.reshape(digits.images.shape[0], -1)
# labels = one_hot.fit_transform(digits.target.reshape(-1, 1))
labels = digits.target.reshape(-1, 1)
train_x, valid_x, train_y, valid_y = train_test_split(feature, labels, test_size=0.333, random_state=0)

train = xgb.DMatrix(train_x, train_y)
valid = xgb.DMatrix(valid_x, valid_y)


# Gridsearch
# parameters = {
#     'max_depth': [5, 10, 15, 20, 25],
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#     'n_estimators': [500, 1000, 2000, 3000, 5000],
#     'min_child_weight': [0, 2, 5, 10, 20],
#     'max_delta_step': [0, 0.2, 0.6, 1, 2],
#     'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
#     'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
#     'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
#
# }

parameters = {
    'max_depth': [5, 10],
    'learning_rate': [0.02, 0.05, 0.1],
    'n_estimators': [2000, 3000],
    'min_child_weight': [0, 2, 5],
    'max_delta_step': [0, 0.2, 0.6],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0, 0.25, 0.5],
    'reg_lambda': [0.4, 0.6, 0.8],
    'scale_pos_weight': [0.4, 0.6, 0.8]

}

xlf = xgb.XGBClassifier(silent=True,
                        objective='multi:softmax',
                        nthread=2,
                        seed=1440,
                        missing=None)

gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=2)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
