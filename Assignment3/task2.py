import numpy as np
from Assignment3.bayes import *
import matplotlib.pyplot as plt


def parse_data(num1, num2, amount_train):
    data = np.load('mnist_all.npz')

    train0 = data[f'train{num1}']
    train1 = data[f'train{num2}']

    test0 = data[f'test{num1}']
    test1 = data[f'test{num2}']

    testLen = len(test0) + len(test1)
    x_train, y_train = gensmallm([train0, train1], [-1, 1], amount_train)
    x_test, y_test = gensmallm([test0, test1], [-1, 1], testLen)

    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)
    return x_train, y_train, x_test, y_test




def naive_Bayes(num1, num2, train_amounts):
    error = []
    for amount in train_amounts:
        x_train, y_train, x_test, y_test = parse_data(num1, num2, amount)

        y_predict = bayespredict(*bayeslearn(x_train, y_train), x_test)
        error.append(np.mean(y_test.reshape(len(y_test), 1) != y_predict))
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

def task2cd(num1, num2):
    x_train, y_train, x_test, y_test = parse_data(num1, num2, 10000)
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    heatMap('ppos', ppos)
    heatMap('pneg', pneg)

    y_predict = bayespredict(allpos, ppos, pneg, x_test)
    y_predict_075 = bayespredict(0.75, ppos, pneg, x_test)

    l = y_predict.shape[0]
    changed_to_minus = np.asarray([1 if y_predict[i] == 1 and y_predict_075[i] == -1 else 0 for i in range(l)])
    changed_from_minus = np.asarray([1 if y_predict[i] == -1 and y_predict_075[i] == 1 else 0 for i in range(l)])

    print(f'The precent of test set example their labels changed from 1 to -1 is {np.mean(changed_to_minus)}')
    print(f'The precent of test set example their labels changed from -1 to 1 is {np.mean(changed_from_minus)}')




def heatMap(title, img):
    plt.imshow(img.reshape(28, 28), cmap='hot')
    plt.title(f'{title} for classification differentiating 0 and 1')
    plt.show()





task2a()
task2cd(0, 1)
task2cd(3, 5)


