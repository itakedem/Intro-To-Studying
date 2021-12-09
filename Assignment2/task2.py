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
    meansError_test = []
    meansError_train = []
    minKeeperTest = []
    maxKeeperTest = []
    minKeeperTrain = []
    maxKeeperTrain = []
    lambda_arr = []
    for n in power:
        lambda_arr = np.append(lambda_arr, l ** n)
        sumMeanI_test = 0
        sumMeanI_train = 0
        currAvg_test_arr = []
        currAvg_train_arr = []
        for i in range(iter):
            _Xtrain, _Ytrain = take_random(x_train, y_train, sample_size)
            w = softsvm(l ** n, _Xtrain, _Ytrain)
            y_test_predict = predict(w, x_test)
            y_train_predict = predict(w, _Xtrain)
            currAvg_test = np.mean(y_test != y_test_predict)
            currAvg_train = np.mean(_Ytrain != y_train_predict)
            sumMeanI_test += currAvg_test
            sumMeanI_train += currAvg_train
            currAvg_test_arr.append(currAvg_test)
            currAvg_train_arr.append(currAvg_train)
        minKeeperTest.append(np.min(currAvg_test_arr))
        maxKeeperTest.append(np.max(currAvg_test_arr))
        minKeeperTrain.append(np.min(currAvg_train_arr))
        maxKeeperTrain.append(np.max(currAvg_train_arr))
        meansError_test = np.append(meansError_test, [sumMeanI_test / iter])
        meansError_train = np.append(meansError_train, [sumMeanI_train / iter])

    return meansError_test, meansError_train, minKeeperTest, maxKeeperTest, minKeeperTrain, maxKeeperTrain,  lambda_arr






power = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
meansError_test, meansError_train, minKeeperTest, maxKeeperTest, minKeeperTrain, maxKeeperTrain, lambda_arr = test_softsvm(power, 10, 10, 100)
plt.errorbar(lambda_arr,
             meansError_train,
             [meansError_train - minKeeperTrain, maxKeeperTrain - meansError_train],
             label='Average train error 100')
plt.errorbar(lambda_arr,
             meansError_test,
             [meansError_test - minKeeperTest, maxKeeperTest - meansError_test],
              label='Average test error 100')

power = [1, 3, 5, 8]
meansError_test, meansError_train, minKeeperTest, maxKeeperTest, minKeeperTrain, maxKeeperTrain, lambda_arr = test_softsvm(power, 1, 10, 1000)
plt.scatter(lambda_arr, meansError_train, color="purple", label='Average train error 1000')
plt.scatter(lambda_arr, meansError_test, color="green", label='Average test error 1000')
plt.xscale('log')
plt.legend()
plt.title("Error over lambda")
plt.xlabel("lambda")
plt.ylabel("Average error")
plt.show()




