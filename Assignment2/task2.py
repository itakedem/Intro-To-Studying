from Assignment2.softsvm import softsvm
import matplotlib.pyplot as plt
import numpy as np

data = np.load('ex2q2_mnist.npz')
x_train = data['Xtrain']
y_train = data['Ytrain']
x_test = data['Xtest']
y_test = data['Ytest']

def take_random(x_list: list, y_list: list, m: int):
    indices = np.random.permutation(x_list.shape[0])
    _x_list = x_list[indices[:m]]
    _y_list = y_list[indices[:m]]
    return _x_list, _y_list



def predict(w, X: np.array):
    return np.sign(X @ w).flatten()


def test_softsvm(power, iter, l, sample_size):
    meansError_test = np.array([])
    meansError_train = np.array([])
    minKeeper = np.array([])
    maxKeeper = np.array([])
    lambda_arr = np.array([])
    for n in power:
        lambda_arr = np.append(lambda_arr, l ** n)
        sumMeanI_test = 0
        sumMeanI_train = 0
        maxError = 0
        minError = 10000
        for i in range(iter):
            _Xtrain, _Ytrain = take_random(x_train, y_train, sample_size)
            w = softsvm(l ** n, _Xtrain, _Ytrain)
            y_test_predict = predict(w, x_test)
            y_train_predict = predict(w, _Xtrain)
            currAvg_test = np.mean(y_test != y_test_predict)
            currAvg_train = np.mean(_Ytrain != y_train_predict)
            sumMeanI_test += currAvg_test
            sumMeanI_train += currAvg_train
            minError = min(minError, currAvg_test, currAvg_train)
            maxError = max(maxError, currAvg_test, currAvg_train)
        minKeeper = np.append(minKeeper, minError)
        maxKeeper = np.append(maxKeeper, maxError)
        meansError_test = np.append(meansError_test, [sumMeanI_test / iter])
        meansError_train = np.append(meansError_train, [sumMeanI_train / iter])

    return meansError_test, meansError_train, minKeeper, maxKeeper, lambda_arr


power = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
meansError_test, meansError_train, minError, maxError, lambda_arr = test_softsvm(power, 10, 10, 100)
plt.semilogx(lambda_arr, meansError_train, color="blue")
plt.semilogx(lambda_arr, meansError_test, color="green")
plt.errorbar(lambda_arr,
             meansError_train,
             [meansError_train - minError, maxError - meansError_train],
             fmt='none', lw=1,
             ecolor='tomato')
plt.errorbar(lambda_arr,
             meansError_test,
             [meansError_test - minError, maxError - meansError_test],
             fmt='none', lw=1,
             ecolor='tomato')

power = [1, 3, 5, 8]
meansError_test, meansError_train, minError, maxError, lambda_arr = test_softsvm(power, 1, 10, 1000)
plt.scatter(lambda_arr, meansError_train, color="yellow")
plt.scatter(lambda_arr, meansError_test, color="brown")
plt.xscale('log')

plt.legend(['Average train error 100', 'Average test error 100', 'Average train error 1000', 'Average test error 1000'])
plt.title("Error over lambda")
plt.xlabel("lambda")
plt.ylabel("Average error")
plt.show()




