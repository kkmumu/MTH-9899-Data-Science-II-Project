## MTH-9899-Data-Science-II-Project

We provide five models to test, but the overall out-of-sample best-performanced model is LightGBM, which has achieved the score of 12.6 bps (weighted R-square). If you want to test other models, please specify model in function predict() in mode 2.

### models:

+ SimpleLinearRegression

+ Ridge
    alpha = 1.35
    Note that linear models are fitted by subset of features.

+ LightGBM
    best_paras = {'learning_rate': 0.05,
                  'num_leaves': 24,
                  'feature_fraction': 0.1,
                  'bagging_fraction': 0.8,
                  'max_depth': 5}
    early_stopping_rounds = 0
    
+ ExtraTrees
    best_paras = {'n_estimators': 150,
                 'min_samples_leaf': 5,
                  'min_samples_split': 4,
                   'criterion': 'mse',
                    'max_depth': 8}
    
+ XGBoost
    best_paras = { 'max_depth': 5,
                    'min_child_weight':5,
                    'n_estimators': 100,
                    'gamma': 0,
                    'learning_rate': 0.1}
