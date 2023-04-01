"""KNN Classification"""
import numpy as np 
import itertools
import statsmodels as stats
import random 

class knn():
    """
    KNN classifier 
    """
    def __init__(self, k: int):
        """
        k: number of neighbors to consider, an integer in [1, n]
        """
        self.k = k 
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        X_train: training data, shape (n, p) 
        y_train: labels of training data, shape (n,)
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        X_test: test data, shape (m, p)
        
        return
        y_hat: predicted labels, shape (m,) 
        """
        # pairwise distance matrix of test data and training data, shape (m, n)
        dist = np.linalg.norm(X_test[:,None,:]-self.X_train[None,:,:],axis=-1)

        # neighbors shape (m, k) 
        neighbors = np.argsort(dist, axis=1)[:,:self.k] 
        y_hat = self.y_train[stats.mode(neighbors,axis=1,keepdims=True)[0]].flatten()
        
        return y_hat


def kfold_split(X: np.ndarray, K: int=5) -> tuple[list, list]: 
    """
    split dataset into K even folds
    
    @param:
    X: features, shape (n, p)
    K: number of folds, default to 5
    
    return:
    train_id: index of training data, length = n*(1-1/K)
    test_id: index of test data, length = n/K
    """
    ids = np.arange(X.shape[0]) 
    random.seed(0)
    random.shuffle(ids)
    folds = np.split(ids, K)
    for i in range(K):
        test_id = list(folds[i]) 
        train_id = [fold for j, fold in enumerate(folds) if j!= i]
        train_id = list(itertools.chain(*train_id))
        yield train_id, test_id 


def kfoldCV(X: np.ndarray, y: np.ndarray, clf, K: int=5) -> tuple[float, float]:
    """
    X: features, shape (n, p) 
    y: labels, shape (n,)
    clf: classifier
    K: number of folds, default to 5
    
    return:
    train_error: misclassification error rate of training, range [0, 1]
    test_error: misclassification error rate of test, range [0, 1]
    """
    train_count, test_count = 0, 0
    for train_id, test_id in kfold_split(X, K):
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        clf.fit(X_train, y_train)
        yhat_train = clf.predict(X_train)
        yhat_test = clf.predict(X_test)
        train_count += np.sum(yhat_train!=y_train) 
        test_count += np.sum(yhat_test!=y_test)
    train_error = train_count/(X.shape[0] * (K-1)) 
    test_error = test_count/X.shape[0]
    
    return train_error, test_error