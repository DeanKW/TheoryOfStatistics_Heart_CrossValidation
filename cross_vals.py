from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

#Implementing cross validation
def perform_k_fold_cross_validation(num_folds, df, response_var, shuffle=False, verbose=False):
    """Performs a single K fold cross validation with a logistic regression classifier

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        shuffle (bool): Whether or not to shuffle the data before making consecutive folds.  Defaults to False
        verbose (bool): Should scores be printed.  Defaults to False

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=None)
    model = LogisticRegression(solver= 'liblinear')

    # X is all predictor variables, y is response variable
    X = df.drop(response_var, axis=1)
    y = df[response_var]

    acc_score = []
    jacc_score = []
    f1_scores = []

    for train_index , test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = y[train_index] , y[test_index]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(y_test, pred_values)
        jacc = jaccard_score(y_test, pred_values)
        f1_sc = f1_score(y_test, pred_values)
        acc_score.append(acc)
        jacc_score.append(jacc)
        f1_scores.append(f1_sc)

    avg_acc, avg_jacc, avg_f1 = calc_scores(acc_score, jacc_score, f1_scores, num_folds, verbose)
    return avg_acc, avg_jacc, avg_f1


#Implementing Multiple Prediction Cross Validation
def perform_MPCV(num_folds, df, response_var, shuffle=False, verbose=False):
    """Performs a single Multiple Predicting cross validation with a logistic regression classifier

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        shuffle (bool): Whether or not to shuffle the data before making consecutive folds.  Defaults to False
        verbose (bool): Should scores be printed.  Defaults to False

    Returns:
        tuple: (avg_acc, avg_jacc, avg_f1) - the average accuracy score, jaccard score, and F1 score

    """
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=None)
    model = LogisticRegression(solver= 'liblinear')

    acc_score = []
    jacc_score = []
    f1_scores = []

    # X is all predictor variables, y is response variable
    X = df.drop(response_var, axis=1)
    y = df[response_var]

    for test_index, train_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = y[train_index] , y[test_index]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(y_test, pred_values)
        jacc = jaccard_score(y_test, pred_values)
        f1_sc = f1_score(y_test, pred_values)

        acc_score.append(acc)
        jacc_score.append(jacc)
        f1_scores.append(f1_sc)

    avg_acc, avg_jacc, avg_f1 = calc_scores(acc_score, jacc_score, f1_scores, num_folds, verbose)
    return avg_acc, avg_jacc, avg_f1

def calc_scores(acc_score_list, jacc_score_list, f1_score_list, num_folds, verbose=False):
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

    if verbose:
        print('accuracy of each fold - {}'.format(acc_score_list))
        print('Avg accuracy : {}'.format(avg_acc_score))
        print()
        print('Jaccard Score of each fold - {}'.format(jacc_score_list))
        print('Avg Jaccard : {}'.format(avg_jacc_score))
        print()
        print('F1 Score of each fold - {}'.format(f1_score_list))
        print('Avg F1 Score : {}'.format(avg_f1_score))

    return avg_acc_score, avg_jacc_score, avg_f1_score

def iterate_cross_validation(num_folds, df, response_var, cross_val_type, num_iter=100, verbose=False, shuffle=True):
    """Performs multiple cross validations with a logistic regression classifier, type selectable

    Args:
        num_folds: the number of folds to use, k = num_folds
        df: Data containing response and predictor variables
        response_var: The variable representing the classification
        cross_val_type: The type of cross validation to perform.  Options include:
            * 'KFCV' - Traditional K-fold Cross Validation
            * 'MPCV' - Multiple Predicting Cross Validation
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
    for i in range(num_iter):
        if cross_val_type=='KFCV':
            acc, jacc, f1 = perform_k_fold_cross_validation(num_folds, df, response_var, shuffle=shuffle, verbose=verbose)
            disp_name = 'Traditional K-Fold Cross Validation'
        elif cross_val_type=='MPCV':
            acc, jacc, f1 = perform_MPCV(num_folds, df, response_var, shuffle=shuffle, verbose=verbose)
            disp_name = 'Multiple Predicting Cross Validation'
        else:
            raise ValueError('cross_val_type must be either CV or MPCV')

        acc_avgs.append(acc)
        jacc_avgs.append(jacc)
        f1_avgs.append(f1)

    mean_acc = sum(acc_avgs)/num_iter
    mean_jacc = sum(jacc_avgs)/num_iter
    mean_f1 = sum(f1_avgs)/num_iter

    print(f'**** {disp_name} ****')
    print(f'Avg accuracy out of {num_iter}: {mean_acc}')
    print(f'Avg jaccard score out of {num_iter}: {mean_jacc}')
    print(f'Avg F1 score out of {num_iter}: {mean_f1}')