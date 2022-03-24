### Author: Shangwen Sun, Ming Fu

from prep import *

def weighted_mse(model, y_true, X):
    r2_weight = 1 / np.array(X['estVol'])
    r2_weight = r2_weight / r2_weight.sum()
    y_pred = model.predict(X)
    return r2_score(y_true, y_pred, sample_weight = r2_weight)

def cv_evaluate(model, X, y, X_test, y_test, how, cv_folds = 5):
    # cross validation
    cv_train_r2, cv_valid_r2 = [], []
    
    if how == "walk_forwarding":
        tscv = TimeSeriesSplit(n_splits = cv_folds)
        splits = tscv.split(X)
        for train_index, test_index in splits:
            X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[test_index]
            y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train_cv, y_train_cv, eval_metric='logloss')
            cv_train_r2 += [weighted_mse(model, y_train_cv, X_train_cv)]
            cv_valid_r2 += [weighted_mse(model, y_valid_cv, X_valid_cv)]
    else:
        n_samples = len(X)
        folds = n_samples // cv_folds
        indices = np.arange(n_samples)

        margin = 0
        for i in range(cv_folds):
            start = i * folds
            stop = start + folds
            temp = int(0.8 * (stop - start)) + start #If you want to change the data ratio of train/Validation, change the 0.8 part.
        
            X_train_cv, X_valid_cv = X.iloc[start: temp], X.iloc[temp + margin: stop]
            y_train_cv, y_valid_cv = y.iloc[start: temp], y.iloc[temp + margin: stop]

            model.fit(X_train_cv, y_train_cv, eval_metric='logloss')
            cv_train_r2 += [weighted_mse(model, y_train_cv, X_train_cv)]
            cv_valid_r2 += [weighted_mse(model, y_valid_cv, X_valid_cv)]
    
    print("Cross Validation Report")
    print(f"cv train r2: {np.array(cv_train_r2).mean()*100}%, cv valid r2: {np.array(cv_valid_r2).mean()*100}%")
    
    return np.mean(cv_valid_r2)
    


def parameter_tuning(my_model, X, y, X_valid, y_valid, X_train_valid, y_train_valid, X_test, y_test, para_test, cv_folds = 5, how = "walk_forwarding"):
    
    # cross validation to select best variables
    keys, values = zip(*para_test.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    weighted_r2_max = -float("inf")
    for p in permutations_dicts:
        print('\n',p)
        my_model.set_params(**p)
        weighted_r2 = cv_evaluate(my_model, X_train_valid, y_train_valid, X_test, y_test, how, cv_folds)
        if weighted_r2 > weighted_r2_max:
            weighted_r2_max = weighted_r2
            best_paras = p
    my_model.set_params(**best_paras)
    print(f"best_paras: {best_paras}")
    
    return best_paras
    
    
    
    
    
