from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score

#Implementing cross validation
def perform_cross_validation(num_folds, df):
    kf = KFold(n_splits=num_folds, random_state=None)
    model = LogisticRegression(solver= 'liblinear')

    acc_score = []
    jacc_score = []

    for train_index , test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = y[train_index] , y[test_index]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(y_test, pred_values)
        jacc = jaccard_score(y_test, pred_values)
        acc_score.append(acc)
        jacc_score.append(jacc)

    avg_acc_score = sum(acc_score)/num_folds
    avg_jacc_score = sum(jacc_score)/num_folds

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print()
    print('Jaccard Score of each fold - {}'.format(jacc_score))
    print('Avg Jaccard : {}'.format(avg_jacc_score))



#Implementing Multiple Prediction Cross Validation
def perform_MPCV(num_folds, df):
    kf = KFold(n_splits=num_folds, random_state=None)
    model = LogisticRegression(solver= 'liblinear')

    acc_score = []
    jacc_score = []

    for test_index, train_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = y[train_index] , y[test_index]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(y_test, pred_values)
        jacc = jaccard_score(y_test, pred_values)
        acc_score.append(acc)
        jacc_score.append(jacc)

    avg_acc_score = sum(acc_score)/num_folds
    avg_jacc_score = sum(jacc_score)/num_folds

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print()
    print('Jaccard Score of each fold - {}'.format(jacc_score))
    print('Avg Jaccard : {}'.format(avg_jacc_score))
