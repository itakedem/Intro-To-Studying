import numpy as np
import math
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def gaussian_kernel(x, x_tag, sigma):
    norm = np.linalg.norm(x - x_tag)
    return math.exp((norm**2) / (-2*sigma))

def gram_matrix(trainX, sigma):
    l = len(trainX)
    gram = np.zeros((l, l))
    for i in range(l):
        for j in range(i, l):
            if i == j:
                gram[i][i] = 1
            else:
                gram[i][j] = gaussian_kernel(trainX[i], trainX[j], sigma)
                gram[j][i] = gram[i][j]
    return gram


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m, d = trainX.shape
    u = matrix(np.concatenate((np.zeros(m), (1 / m) * np.ones(m))))
    gram = gram_matrix(trainX, sigma)
    H = np.block([[2 * l * gram, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]])
    H = H + (1e-6) * np.eye(m + m)
    H = sparse(matrix(H))
    A = np.block([[np.zeros((m, m)), np.eye(m)], [np.diag(trainy) @ gram.T, np.eye(m)]])
    A = sparse(matrix(A))
    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))
    sol = solvers.qp(H, u, -A, -v)
    sol = np.asarray(sol["x"])
    return sol[:m]


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)
    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"



if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
