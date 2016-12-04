import numpy as np
import scipy.sparse as sc
from numpy import hstack, savetxt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from collections import Counter
import scipy

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score, precision_score


def main():

    # Preparing train data
    dataset = np.genfromtxt('census-income.data.gz',delimiter=',',usemask=False,dtype=str)
    target = [x[41] for x in dataset]
    target = [0 if y==' - 50000.' else 1 for y in target]
    
    data = [x[0] for x in dataset]
    le = preprocessing.LabelEncoder()
    data = le.fit_transform(data)
    for i in range(1,41):
        data = np.column_stack((data, le.fit_transform([x[i-1] for x in dataset])))
	
    # Preparing test data
    dataset = np.genfromtxt('census-income.test.gz',delimiter=',',usemask=False,dtype=str)
    y_test = [x[41] for x in dataset]
    y_test = [0 if y==' - 50000.' else 1 for y in y_test]
    
    X_test = [x[0] for x in dataset]
    X_test = le.fit_transform(X_test)
    for i in range(1,41):
        X_test = np.column_stack((X_test, le.fit_transform([x[i-1] for x in dataset])))

    print(data.shape)
    print(len(target))
    print(X_test.shape)
    print(len(y_test))

    X_first_train, X_second_train, y_first_train, y_second_train = train_test_split(data, target, test_size=0.5, random_state=35)
    X_first_train = sc.csc_matrix(X_first_train)
    X_test = sc.csc_matrix(X_test)
    X_second_train = sc.csc_matrix(X_second_train) 

    # Step 1: Building a Random Forest
    initial_rf = RandomForestClassifier(n_estimators=10, max_features = 'log2', min_samples_split = 4000, min_samples_leaf=2000, n_jobs=-1)
    initial_rf.fit(X_first_train, y_first_train)
    n_trees = len(initial_rf.estimators_)


    print("initial data dimension")
    print(X_first_train.shape)
    print("ROC AUC:")
    print(roc_auc_score(y_test, [p[1] for p in initial_rf.predict_proba(X_test)]))
    print("F1 score:")
    print(f1_score(y_test, initial_rf.predict(X_test), average='binary'))
    print("Accuracy:")
    print(accuracy_score(y_test, initial_rf.predict(X_test)))
    print("Precision:")
    print(precision_score(y_test, initial_rf.predict(X_test), average='binary'))

    # Step 2: Running on Test Data, Generating New Dataset 
    # Step 3: Applying Other Classifiers to the New Data
    vectorizer = [CountVectorizer(tokenizer=lambda doc: doc, lowercase=False) for i in range(n_trees)]
    train_data = create_new_data(vectorizer, initial_rf, X_second_train, True)
    test_data = create_new_data(vectorizer, initial_rf, X_test, False)
    print("final data dimension")
    print(train_data.shape)

    res = list()
    res.append(process_final_rf_classifier(train_data, y_second_train, test_data, y_test, 1000,'log2',2046))

    final_cls = LogisticRegression(C=0.1,penalty="l1", class_weight="auto",n_jobs=-1)
    final_cls.fit(train_data, y_second_train)
    print("final ROC AUC with logistic regression")
    print(roc_auc_score(y_test, [p[1] for p in final_cls.predict_proba(test_data)]))
    print("F1 score:")
    print(f1_score(y_test, final_cls.predict(test_data), average='binary'))
    print("Precision:")
    print(precision_score(y_test, final_cls.predict(test_data), average='binary'))

    train_data = create_new_data_v(vectorizer, initial_rf, X_second_train, True)
    test_data = create_new_data_v(vectorizer, initial_rf, X_test, False)

    print("final data dimension")
    print(train_data.shape)

    res = list()
    res.append(process_final_rf_classifier(train_data, y_second_train, test_data, y_test, 1000,'log2',2046))
    
    final_cls = LogisticRegression(C=0.1,penalty="l1", class_weight="auto",n_jobs=-1)
    final_cls.fit(train_data, y_second_train)
    print("final answer with logistic regression")
    print(roc_auc_score(y_test, [p[1] for p in final_cls.predict_proba(test_data)]))
    print("F1 score:")
    print(f1_score(y_test, final_cls.predict(test_data), average='binary'))
    print("Accuracy:")
    print(accuracy_score(y_test, final_cls.predict(test_data)))
    print("Precision:")
    print(precision_score(y_test, final_cls.predict(test_data), average='binary'))
    
def process_final_rf_classifier(train_data, y_normal_run, test_data, y_test, n_trees, n_features,n_split):
    final_cls = RandomForestClassifier(n_estimators=n_trees, max_features = n_features, min_samples_split = n_split, min_samples_leaf=n_split/2, n_jobs=-1)
    final_cls.fit(train_data, y_normal_run)
    auc = roc_auc_score(y_test, [p[1] for p in final_cls.predict_proba(test_data)])
    print("Final ROC AUC with Random Forest of %0.0f trees, with max_features of %s, min_samples_split %0.0f and min_samples_leaf of %0.0f: %s" % (n_trees, n_features, n_split,n_split/2,auc))
    print("F1 score:")
    print(f1_score(y_test, final_cls.predict(test_data), average='binary'))
    print("Accuracy:")
    print(accuracy_score(y_test, final_cls.predict(test_data)))
    print("Precision:")
    print(precision_score(y_test, final_cls.predict(test_data), average='binary'))
    return auc

def produce_probabilities(trees, X):
    probas = []
    for tree in trees:
        probas.append([p[1] for p in tree.predict_proba(X)])    
    return np.array(probas).transpose()

def produce_indices_v(vectorizer, rf, X, train = True):
    new_leaf_indices = rf.apply(X)
    n = new_leaf_indices.shape[0]
    if train:
        temp = [v.fit_transform([str(index) for index in new_leaf_indices[:]]).toarray() for i,v in enumerate(vectorizer)]
        new_indices_vectorizer = np.hstack(temp)
    else:
        new_indices_vectorizer = np.hstack([v.transform([str(index) for index in new_leaf_indices[:]]).toarray() for i,v in enumerate(vectorizer)])
    return new_indices_vectorizer

def produce_indices(vectorizer, rf, X, train = True):
    new_leaf_indices = rf.apply(X)
    return new_leaf_indices

def create_new_data(vectorizer, rf, X, train = True):    
    I = produce_indices(vectorizer, rf, X, train)
    P = produce_probabilities(rf.estimators_, X)
    return np.hstack((I, P))    

def create_new_data_v(vectorizer, rf, X, train = True):    
    I = produce_indices_v(vectorizer, rf, X, train)
    P = produce_probabilities(rf.estimators_, X)
    return np.hstack((I, P))    

if __name__=="__main__":
    main()



