import matplotlib.pyplot as plt
import random
import numpy as np
from nearest_neighbour import gensmallm, learnknn, predictknn


def testRepeaterOverTrainSize():
    k = 1
    data = np.load('mnist_all.npz')
    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    testLen = max(len(test1), len(test3), len(test4), len(test6))
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
    for i in range(int(size / 5)):
        options = [1, 3, 4, 6]
        options.remove(y[i])
        y[i] = random.choice(options)


def testRepeaterOverK(corrupted):
    data = np.load('mnist_all.npz')
    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    testLen = max(len(test1), len(test3), len(test4), len(test6))
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
    meansErrorTrain, testSizeTrain, minError, maxError = testRepeaterOverTrainSize()
    plt.plot(testSizeTrain, meansErrorTrain, color="blue")
    plt.plot(testSizeTrain, minError, color="green")
    plt.plot(testSizeTrain, maxError, color="red")
    plt.errorbar(testSizeTrain,
                 meansErrorTrain,
                 [meansErrorTrain - minError, maxError - meansErrorTrain],
                 fmt='ok', lw=1,
                 ecolor='tomato')
    plt.legend(["Average test error", "Minimum Error", "Maximum Error"])
    plt.title("Error over train size")
    plt.xlabel("Train size")
    plt.ylabel("Average test error")
    plt.show()

    meansErrorK, testSizeK = testRepeaterOverK(False)
    plt.plot(testSizeK, meansErrorK, color="blue")
    plt.legend(["Average test error"])
    plt.title("Error over K")
    plt.xlabel("K")
    plt.ylabel("Average test error")
    plt.show()

    meansErrorK, testSizeK = testRepeaterOverK(True)
    plt.plot(testSizeK, meansErrorK, color="blue")
    plt.legend(["Average test error"])
    plt.title("Error over corrupted K")
    plt.xlabel("K")
    plt.ylabel("Average test error")
    plt.show()
