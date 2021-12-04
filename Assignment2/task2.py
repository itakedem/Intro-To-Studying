from Assignment2.softsvm import softsvm
import matplotlib.pyplot as plt
import numpy as np


def take_random(x_list: list, y_list: list, m: int):
    random_i = np.random.randint(0, len(y_list), m)
    x_list_rnd = np.asarray([x_list[i] for i in random_i])
    y_list_rnd = np.asarray([y_list[i] for i in random_i])
    return x_list_rnd, y_list_rnd



def predict(w, X: np.array):
    return np.array([np.sign(x @ w) for x in X])


def test_softsvm(power, iter, l, sample_size):
    data = np.load('ex2q2_mnist.npz')
    x_train = data['Xtrain']
    y_train = data['Ytrain']
    x_test = data['Xtest']
    y_test = data['Ytest']
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
            x_train, y_train = take_random(x_train, y_train, sample_size)
            w = softsvm(l ** n, x_train, y_train)
            y_test_predict = predict(w, x_test)
            y_train_predict = predict(w, x_train)
            currAvg_test = np.mean(y_test != y_test_predict)
            currAvg_train = np.mean(y_train != y_train_predict)
            sumMeanI_test += currAvg_test
            sumMeanI_train += currAvg_train
            minError = min(minError, currAvg_test, currAvg_train)
            maxError = max(maxError, currAvg_test, currAvg_train)
        minKeeper = np.append(minKeeper, minError)
        maxKeeper = np.append(maxKeeper, maxError)
        meansError_test = np.append(meansError_test, [sumMeanI_test / iter])
        meansError_train = np.append(meansError_train, [sumMeanI_train / iter])

    return meansError_test, meansError_train, minKeeper, maxKeeper, lambda_arr


power = np.arange(1, 11)
meansError_test, meansError_train, minError, maxError, lambda_arr = test_softsvm(power, 10, 10, 100)
plt.semilogx(lambda_arr, meansError_train, color="blue", legend='Average train error')
plt.semilogx(lambda_arr, meansError_test, color="green", legend='Average test error')
plt.errorbar(lambda_arr,
             meansError_train,
             [meansError_train - minError, maxError - meansError_train],
             fmt='ok', lw=1,
             ecolor='tomato')
plt.errorbar(lambda_arr,
             meansError_test,
             [meansError_test - minError, maxError - meansError_test],
             fmt='ok', lw=1,
             ecolor='tomato')
plt.legend()
plt.title("Error over lambda")
plt.xlabel("lambda")
plt.ylabel("Average error")
plt.show()
