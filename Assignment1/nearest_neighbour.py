import random

import matplotlib.pyplot as plt
import numpy as np
import time


class Classifier:
    def __init__(self, k, xTrain, yTrain):
        self.k = k
        self.xTrain = xTrain
        self.yTrain = yTrain

    def clasify(self, x):
        dist = np.array([(np.linalg.norm(self.xTrain[i] - x), self.yTrain[i]) for i in range(len(self.xTrain))])
        dist = dist[dist[:, 0].argsort()]  # sort by the first column
        topK = dist[:self.k]  # get only k first rows
        topK = topK[:, 1]  # get the second column
        values, counts = np.unique(topK, return_counts=True)
        ind = np.argmax(counts)  # together they bring the index of the most frequent value
        return values[ind]


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """
    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return Classifier(k, x_train, y_train)


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    yPrediction = np.array([classifier.clasify(x) for x in x_test])
    # for x in x_test:
    #     np.append(yPrediction, classifier.clasify(x))
    return yPrediction.reshape((len(yPrediction), 1))


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])
    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def verySimpleTest():
    arrSorted = np.array([1, 1, 5, 3, 1, 9, 2, 2, 2, 1, 2, 3, 1])
    values, counts = np.unique(arrSorted, return_counts=True)
    ind = np.argmax(counts)

    print("expected: \n", 2, "\ngot: \n", values[ind])


def testRepeaterOverTrainSize():
    k = 1
    data = np.load('mnist_all.npz')
    testLen = 4085
    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    testSize = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], testLen)

    meansError = np.array([])
    minKeeper = np.array([])
    maxKeeper = np.array([])
    for n in testSize:
        sumMeanI = 0
        maxError = 0
        minError = 10000
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], n)
            classifier = learnknn(k, x_train, y_train)
            preds = predictknn(classifier, x_test)
            preds = preds.reshape(testLen, )
            currAvg = np.mean(y_test != preds)
            sumMeanI += currAvg
            if currAvg < minError:
                minError = currAvg
            if currAvg > maxError:
                maxError = currAvg
        minKeeper = np.append(minKeeper, minError)
        maxKeeper = np.append(maxKeeper, maxError)
        meansError = np.append(meansError, [sumMeanI / 10.0])

    return meansError, testSize, minKeeper, maxKeeper


def corrupt(y):
    size = len(y)
    indexes = [random.randint(0, size-1) for i in range(int(size/5))]
    for i in indexes:
        options = [1, 3, 4, 6]
        options.remove(y[i])
        y[i] = random.choice(options)


def testRepeaterOverK(corrupted):
    data = np.load('mnist_all.npz')
    testLen = 4085
    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], testLen)

    meansError = np.array([])
    for k in range(1, 12):
        sumMeanI = 0
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], 100)
            if corrupted:
                corrupt(y_train)
                corrupt(y_test)
            classifier = learnknn(k, x_train, y_train)
            preds = predictknn(classifier, x_test)
            preds = preds.reshape(testLen, )
            sumMeanI += np.mean(y_test != preds)

        meansError = np.append(meansError, [sumMeanI / 10.0])

    return meansError, np.arange(1, 12)


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # meansErrorTrain, testSizeTrain, minError, maxError = testRepeaterOverTrainSize()
    # plt.plot(testSizeTrain, meansErrorTrain, color="blue")
    # plt.plot(testSizeTrain, minError, color="green")
    # plt.plot(testSizeTrain, maxError, color="red")
    # plt.errorbar(testSizeTrain,
    #              meansErrorTrain,
    #              [meansErrorTrain - minError, maxError - meansErrorTrain],
    #              fmt='ok', lw=1,
    #              ecolor='tomato')
    # plt.legend(["Average test error", "Minimum Error", "Maximum Error"])
    # plt.title("Error over train size")
    # plt.xlabel("Train size")
    # plt.ylabel("Average test error")
    # plt.show()
    # plt.close()
    #
    # meansErrorK, testSizeK = testRepeaterOverK(False)
    # plt.plot(testSizeK, meansErrorK, color="blue")
    # plt.legend(["Average test error"])
    # plt.title("Error over K")
    # plt.xlabel("K")
    # plt.ylabel("Average test error")
    # plt.show()
    # plt.close()
    #

    meansErrorK, testSizeK = testRepeaterOverK(True)
    plt.plot(testSizeK, meansErrorK, color="blue")
    plt.legend(["Average test error"])
    plt.title("Error over corrupted K")
    plt.xlabel("K")
    plt.ylabel("Average test error")
    plt.show()
    plt.close()




