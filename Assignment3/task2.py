import numpy as np
from Assignment3.bayes import *
import matplotlib.pyplot as plt


def parse_data(num1, num2, amount_train):
    data = np.load('mnist_all.npz')
    train0 = np.where(data[f'train{num1}'] > 128, 1, 0)
    train1 = np.where(data[f'train{num2}'] > 128, 1, 0)

    test0 = np.where(data[f'test{num1}'] > 128, 1, 0)
    test1 = np.where(data[f'test{num2}'] > 128, 1, 0)

    testLen = max(len(test0), len(test1))
    x_test, y_test = gensmallm([test0, test1], [-1, 1], testLen)
    x_train, y_train = gensmallm([train0, train1], [-1, 1], amount_train)
    return x_train, y_train, x_test, y_test




def naive_Bayes(num1, num2, train_amounts):
    error = []
    for amount in train_amounts:
        x_train, y_train, x_test, y_test = parse_data(num1, num2, amount)

        y_predict = bayespredict(*bayeslearn(x_train, y_train), x_test)
        error.append(np.mean(y_test != y_predict))
    return error

def task2a():
    train_amounts = [i for i in range(1000, 11000, 1000)]
    error01 = naive_Bayes(0, 1, train_amounts)
    error35 = naive_Bayes(3, 5, train_amounts)
    plt.plot(train_amounts, error01, label='error 0,1')
    plt.plot(train_amounts, error35, label='error 3,5')
    plt.legend()
    plt.title("naive-Bayes error")
    plt.xlabel("amount of train data")
    plt.ylabel("error")
    plt.show()


task2a()



