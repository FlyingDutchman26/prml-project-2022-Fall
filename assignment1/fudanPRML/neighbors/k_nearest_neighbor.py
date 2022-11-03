import paddle
import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = paddle.to_tensor(X,dtype = 'float32')
        self.y_train = paddle.to_tensor(y,dtype = 'int32')

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)


    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        """
        X = paddle.to_tensor(X,dtype = 'float32')
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = paddle.zeros((num_test, num_train))
        dists = paddle.sqrt(-2*paddle.matmul(X, self.X_train.t()) + paddle.sum(paddle.square(self.X_train),
                        axis=1) + paddle.sum(paddle.square(X), axis=1,keepdim=True))
        return dists.numpy()

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        self.y_train = self.y_train.numpy()
        for i in range(num_test):
            closest_y = []
            indices = np.argsort(dists[i, :])[0:k]
            closest_y = self.y_train[indices]
            countlist = np.bincount(closest_y)
            y_pred[i] = np.argmax(countlist)

        return y_pred
