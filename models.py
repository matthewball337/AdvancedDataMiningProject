from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from utils import get_average_f1, create_X_y
import pandas as pd

'''
References:

1) Hyperparameter tuning reference: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
2) Other hyperparameter tuning reference: https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide#bayesianoptimization
3) Linear Discriminant Analysis hyperparameter tuning: https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
4) Some pieces of code reused from Fall 2022 Data Mining project

'''

#This creates a Ridge Classifier model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs
def ridge_classifier_accuracy(df, tuning, originalRun):
    if tuning:
        if originalRun:
            X, y = create_X_y(df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
            param_grid_rc = {
                'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
            rc_tuned = RidgeClassifier()
            grid_search_rc = GridSearchCV(estimator=rc_tuned, param_grid=param_grid_rc, n_jobs=-1, cv=3)
            grid_search_rc.fit(X_train, Y_train)
            print(grid_search_rc.best_params_)
            model = RidgeClassifier(**grid_search_rc.best_params_)
            params = grid_search_rc.best_params_
        else:
            params = dict(alpha=0.1)
            print(params)
            model = RidgeClassifier(**params)
    else:
        model = RidgeClassifier()
        params = {}
        
    return get_average_f1(model, df), params

        
#This creates a Naive Bayes model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs

def naive_bayes_accuracy(df):
    model = GaussianNB()
    return get_average_f1(model,df), dict()
    
#This creates the Linear Discriminative Analysis model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs

def lda_accuracy(df, tuning, originalRun):
    if tuning:
        if originalRun:
            X, y = create_X_y(df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
            param_grid_lda = {
                'solver': ['svd', 'lsqr'],
            }
            lda_tuned = LinearDiscriminantAnalysis()
            grid_search_lda = GridSearchCV(estimator=lda_tuned, param_grid=param_grid_lda, cv=3)
            grid_search_lda.fit(X_train, Y_train)
            print(grid_search_lda.best_params_)
            model = LinearDiscriminantAnalysis(**grid_search_lda.best_params_)
            params = grid_search_lda.best_params_
        else:
            params = dict(solver='svd')
            print(params)
            model = LinearDiscriminantAnalysis(**params)
    else:
        model = LinearDiscriminantAnalysis()
        params = {}
        
    return get_average_f1(model, df), params
    
#This creates the Logistic Regression model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs

def logistic_regression_accuracy(df, tuning, originalRun):
    if tuning:
        if originalRun:
            X, y = create_X_y(df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
            param_grid_lr = {
                'Cs': [100, 10, 1],
                'cv': [3],
                'penalty': ['l2'],
                'scoring': ['f1'],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight': ['balanced'],
                'n_jobs': [-1],
            }
            lr_tuned = LogisticRegressionCV()
            grid_search_lr = GridSearchCV(estimator=lr_tuned, param_grid=param_grid_lr, cv=3)
            grid_search_lr.fit(X_train, Y_train)
            print(grid_search_lr.best_params_)
            model = LogisticRegressionCV(**grid_search_lr.best_params_)
            params = grid_search_lr.best_params_
        else:
            params = dict(Cs=100,class_weight='balanced',cv=3,n_jobs=-1,penalty='l2',scoring='f1',solver='lbfgs')
            print(params)
            model = LogisticRegressionCV(**params)
    else:
        model = LogisticRegressionCV(cv=3, scoring='f1', class_weight='balanced', n_jobs=-1)
        params = dict(cv=3,scoring='f1',class_weight='balanced',n_jobs=-1)
        
    return get_average_f1(model, df), params
    
#This creates the Bagging Algorithm model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs
def bagging_accuracy(df, tuning, originalRun):
    if tuning:
        if originalRun:
            X, y = create_X_y(df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
            param_grid_bag = {
                'base_estimator': [GradientBoostingClassifier(max_depth=3,min_samples_leaf=3,n_estimators=300)],
                'n_estimators': [10, 100, 1000]
            }
            bag_tuned = BaggingClassifier()
            grid_search_bag = GridSearchCV(estimator=bag_tuned, param_grid=param_grid_bag, cv=3)
            grid_search_bag.fit(X_train, Y_train)
            print(grid_search_bag.best_params_)
            model = BaggingClassifier(**grid_search_bag.best_params_)
            params = grid_search_bag.best_params_
        else:
            params=dict(base_estimator=GradientBoostingClassifier(max_depth=3,min_samples_leaf=3,n_estimators=300),n_estimators=10)
            print(params)
            model = BaggingClassifier(**params)
    else:
        model = BaggingClassifier(GradientBoostingClassifier(max_depth=3,min_samples_leaf=3,n_estimators=300))
        params = dict(base_estimator=GradientBoostingClassifier(max_depth=3,min_samples_leaf=3,n_estimators=300))
    return get_average_f1(model, df), params
    
#This creates the Gradient Boosting model with or without hypertuning,
#splits the training/test data and averages the f1 score over 10 runs
def gradient_boosting_accuracy(df, tuning, originalRun):
    print('Evaluating gradient boosting model')
    if tuning:
        if originalRun:
            X, y = create_X_y(df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
            param_grid_gb = {
                'n_estimators': [300, 500],
                'max_depth': [3, 5],
                'min_samples_leaf': [2, 3]
            }
            gb_tuned = GradientBoostingClassifier()
            grid_search_gb = GridSearchCV(estimator=gb_tuned, param_grid=param_grid_gb, cv=3)
            grid_search_gb.fit(X_train, Y_train)
            print(grid_search_gb.best_params_)
            model = GradientBoostingClassifier(**grid_search_gb.best_params_)
            params = grid_search_gb.best_params_
        else:
            params=dict(max_depth=3,min_samples_leaf=3,n_estimators=300)
            print(params)
            model = GradientBoostingClassifier(**params)
    else:
        model = GradientBoostingClassifier()
        params = {}
    return get_average_f1(model,df), params