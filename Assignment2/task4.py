from Assignment2.softsvm import softsvm
import matplotlib.pyplot as plt
import numpy as np
from Assignment2.softsvmrbf import softsvmbf, gram_matrix, gaussian_kernel


data = np.load('ex2q4_data.npz')
x_train = data['Xtrain']
y_train = data['Ytrain']
x_test = data['Xtest']
y_test = data['Ytest']




def plot_train_set():
    negative_train = x_train[y_train.flatten() == -1]
    positive_train = x_train[y_train.flatten() == 1]

    plt.scatter(negative_train[:, 0], negative_train[:, 1], color="red", label='-1')
    plt.scatter(positive_train[:, 0], positive_train[:, 1], color="blue", label='1')
    plt.legend()
    plt.title(r'Train Set Points in ${R}^2$')
    plt.show()


def mrbf_error(alpha, xTrain, xTest, yTest, sigma):
    y_predict = []
    for x in xTest:
        sum = 0
        for i in range(len(alpha)):
            if alpha[i] == 0:
                continue
            sum += alpha[i] * gaussian_kernel(xTrain[i], x, sigma)
        y_predict.append(np.sign(sum)[0])
    return np.mean(np.asarray(y_predict) != yTest.flatten())



def svm_error(w, xTest, yTest):
    y_predict = np.sign(xTest @ w).flatten()
    return np.mean(yTest != y_predict)


def fold_cross_validation(k, flag):
    lambda_arr = [1, 10, 100]
    sigma_arr = [0.001, 0.5, 1]
    if flag:
        combination = [[l, s] for l in lambda_arr for s in sigma_arr]
    else:
        combination = lambda_arr
    split_x_train = np.asarray(np.split(x_train, k))
    split_y_train = np.asarray(np.split(y_train, k))

    comb_err = []
    for c in combination:
        err = []
        for i in range(k):
            sub_x_train = resizeArray(np.delete(split_x_train, i, 0))
            sub_y_train = resizeArray(np.delete(split_y_train, i, 0))
            split_x_test = split_x_train[i]
            split_y_test = split_y_train[i]
            if flag:
                alpha = softsvmbf(c[0], c[1], sub_x_train, sub_y_train)
                curr_err = mrbf_error(alpha, sub_x_train, split_x_test, split_y_test, c[1])
            else:
                w = softsvm(c, sub_x_train, sub_y_train)
                curr_err = svm_error(w, split_x_test, split_y_test)
            err.append(curr_err)
        comb_err.append(np.mean(err))
    combo = combination[np.argmin(np.asarray(comb_err))]
    return fullValidation(combo, flag)


def fullValidation(combo, flag):
    if flag:
        alpha = softsvmbf(combo[0], combo[1], x_train, y_train)
        err = mrbf_error(alpha, x_train, x_test, y_test, combo[1])
    else:
        w = softsvm(combo, x_train, y_train)
        err = svm_error(w, x_test, y_test)
    return err, combo


def resizeArray(A: np.ndarray):
    a, b, c = A.shape
    return A.reshape((a*b, c))

err, combo = fold_cross_validation(5, 1)
print("SoftSVMrbf results are:")
print("The optimal combination (lambda, sigma) is ", combo)
print("The optimal error is ", err)

print()

err, combo = fold_cross_validation(5, 0)
print("SoftSVM results are:")
print("The optimal lambda is ", combo)
print("The optimal error is ", err)
