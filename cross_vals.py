from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, roc_auc_score

#Implementing cross validation
def perform_k_fold_cross_validation(num_folds, df, response_var, stratified=False, shuffle=False, verbose=False):
    """Performs a single K fold cross validation with a logistic regression classifier

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        stratified (bool): Whether to ensure stratified folds are used, preserving the percentage of samples for each class
        shuffle (bool): Whether or not to shuffle the data before making consecutive folds.  Defaults to False
        verbose (bool): Should scores be printed.  Defaults to False

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    if stratified:
        if num_folds==len(df):
            raise ValueError('It appears you are trying to do a Stratified LOO, that isn\'t possible')
        kf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=None)
    else:
        kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=None)

    model = LogisticRegression(solver= 'liblinear')

    # X is all predictor variables, y is response variable
    X = df.drop(response_var, axis=1)
    y = df[response_var]

    acc_score = []
    jacc_score = []
    f1_scores = [] 
    roc_auc_scores = []

    if stratified:
        for train_index , test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train, y_test = y[train_index] , y[test_index]

            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(y_test, pred_values)
            jacc = jaccard_score(y_test, pred_values)
            f1_sc = f1_score(y_test, pred_values)
            roc_auc = roc_auc_score(y_test, pred_values)

            acc_score.append(acc)
            jacc_score.append(jacc)
            f1_scores.append(f1_sc)
            roc_auc_scores.append(roc_auc)
    # TODO: Make this prettier.  Possibly split out LOO and then make inner part of for loop a nested func
    else:
        for train_index , test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train, y_test = y[train_index] , y[test_index]

            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(y_test, pred_values)
            acc_score.append(acc)

            # In LOO, only testing on one piece of data at a time, thus, jaccard score and f1 score
            # are not defined
            if len(y_test) > 1:
                jacc = jaccard_score(y_test, pred_values)
                f1_sc = f1_score(y_test, pred_values)
                roc_auc = roc_auc_score(y_test, pred_values)

                jacc_score.append(jacc)
                f1_scores.append(f1_sc)
                roc_auc_scores.append(roc_auc)

    avg_acc, avg_jacc, avg_f1, avg_roc_auc = calc_scores(acc_score, jacc_score, f1_scores, roc_auc_scores, 
                                                         num_folds, verbose)
    return avg_acc, avg_jacc, avg_f1, avg_roc_auc

#Implementing Multiple Prediction Cross Validation
def perform_MPCV(num_folds, df, response_var, stratified=False, shuffle=False, verbose=False):
    """Performs a single Multiple Predicting cross validation with a logistic regression classifier

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        stratified (bool): Whether to ensure stratified folds are used, preserving the percentage of samples for each class
        shuffle (bool): Whether or not to shuffle the data before making consecutive folds.  Defaults to False
        verbose (bool): Should scores be printed.  Defaults to False

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=None)
    else:
        kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=None)
    model = LogisticRegression(solver= 'liblinear')

    acc_score = []
    jacc_score = []
    f1_scores = []
    roc_auc_scores = []

    # X is all predictor variables, y is response variable
    X = df.drop(response_var, axis=1)
    y = df[response_var]

    if stratified:
        for test_index, train_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train, y_test = y[train_index] , y[test_index]

            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(y_test, pred_values)
            jacc = jaccard_score(y_test, pred_values)
            f1_sc = f1_score(y_test, pred_values)
            roc_auc = roc_auc_score(y_test, pred_values)

            acc_score.append(acc)
            jacc_score.append(jacc)
            f1_scores.append(f1_sc)
            roc_auc_scores.append(roc_auc)
    # TODO: Put this in a nested function so don't have copying and pasting
    else:
        for test_index, train_index in kf.split(X):
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train, y_test = y[train_index] , y[test_index]
    
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(y_test, pred_values)
            jacc = jaccard_score(y_test, pred_values)
            f1_sc = f1_score(y_test, pred_values)
            roc_auc = roc_auc_score(y_test, pred_values)
    
            acc_score.append(acc)
            jacc_score.append(jacc)
            f1_scores.append(f1_sc)
            roc_auc_scores.append(roc_auc)

    avg_acc, avg_jacc, avg_f1, avg_roc_auc = calc_scores(acc_score, jacc_score, f1_scores, roc_auc_scores, num_folds, verbose)
    return avg_acc, avg_jacc, avg_f1, avg_roc_auc

def calc_scores(acc_score_list, jacc_score_list, f1_score_list, roc_auc_list, num_folds, verbose=False):
    """Given a list of each fold's scores, calculates the average

    Args:
        acc_score_list (list[float]): The list of accuracy scores
        jacc_score_list (list[float]): The list of Jaccard scores
        f1_score_list (list[float]): The list of F1 scores
        num_folds (int): The number of folds used
        verbose (bool): Should scores be printed

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    avg_acc_score = sum(acc_score_list)/num_folds
    avg_jacc_score = sum(jacc_score_list)/num_folds
    avg_f1_score = sum(f1_score_list)/num_folds
    avg_roc_auc_score = sum(roc_auc_list)/num_folds

    if verbose:
        print('accuracy of each fold - {}'.format(acc_score_list))
        print('Avg accuracy : {}'.format(avg_acc_score))
        print()
        print('Jaccard Score of each fold - {}'.format(jacc_score_list))
        print('Avg Jaccard : {}'.format(avg_jacc_score))
        print()
        print('F1 Score of each fold - {}'.format(f1_score_list))
        print('Avg F1 Score : {}'.format(avg_f1_score))
        print()
        print('ROC AUC of each fold - {}'.format(roc_auc_list))
        print('Avg ROC AUC : {}'.format(avg_roc_auc_score))

    return avg_acc_score, avg_jacc_score, avg_f1_score, avg_roc_auc_score

def iterate_cross_validation(num_folds, df, response_var, cross_val_type, stratified=False, num_iter=100, verbose=False, shuffle=True):
    """Performs multiple cross validations with a logistic regression classifier, type selectable

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        cross_val_type: The type of cross validation to perform.  Options include:
            * 'LOO' - Leave-one-out 
            * 'KFCV' - Traditional K-fold Cross Validation
            * 'MPCV' - Multiple Predicting Cross Validation
        stratified (bool): Whether to ensure stratified folds are used, preserving the percentage of samples for each class
        num_iter  : The number of times to perform the cross validation.  Defaults to 100
        verbose (bool): Should individual iteration scores be printed, used for debugging.  Defaults to False
        shuffle (bool): Whether or not to shuffle the data before making consecutive folds.  Defaults to True
            THIS SHOULD ALWAYS BE TRUE UNLESS DEBUGGING
        

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    acc_avgs = []
    jacc_avgs = []
    f1_avgs = []
    auc_avgs = []
    if cross_val_type == 'LOO':
        print('WARNING: LOO is not fully implemented.  Scoring incomplete')
    if cross_val_type=='LOO' and num_iter != 1:
        print('LOO is deterministic.  Setting num_iter to 1')
        num_iter=1
    for i in range(num_iter):
        if cross_val_type=='KFCV':
            acc, jacc, f1, auc = perform_k_fold_cross_validation(num_folds, df, response_var, stratified=stratified, shuffle=shuffle, verbose=verbose)
            disp_name = 'K-Fold Cross Validation'
        elif cross_val_type=='MPCV':
            acc, jacc, f1, auc = perform_MPCV(num_folds, df, response_var, stratified=stratified, shuffle=shuffle, verbose=verbose)
            disp_name = 'Multiple Predicting Cross Validation'
        elif cross_val_type=='LOO':
            num_folds = len(df)
            acc, jacc, f1, auc = perform_k_fold_cross_validation(num_folds, df, response_var, stratified=False, shuffle=shuffle, verbose=verbose)
            disp_name = 'Leave-One-Out Cross Validation'
        else:
            raise ValueError('cross_val_type must be either KFCV, LOO, or MPCV')

        if stratified and cross_val_type != 'LOO':
            disp_name = 'Stratified '+ disp_name
        acc_avgs.append(acc)
        jacc_avgs.append(jacc)
        f1_avgs.append(f1)
        auc_avgs.append(auc)

    mean_acc = sum(acc_avgs)/num_iter
    mean_jacc = sum(jacc_avgs)/num_iter
    mean_f1 = sum(f1_avgs)/num_iter
    mean_auc = sum(auc_avgs)/num_iter

    print(f'**** {disp_name} ****')
    print(f'Avg accuracy out of {num_iter} iterations: {mean_acc}')
    print(f'Avg jaccard score out of {num_iter} iterations: {mean_jacc}')
    print(f'Avg F1 score out of {num_iter} iterations: {mean_f1}')
    print(f'Avg ROC AUC out of {num_iter} iterations: {mean_auc}')