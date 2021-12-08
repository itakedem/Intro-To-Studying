from Assignment2.softsvm import softsvm
import matplotlib.pyplot as plt
import numpy as np
from Assignment2.softsvmrbf import softsvmbf


data = np.load('ex2q2_mnist.npz')
x_train = data['Xtrain']
y_train = data['Ytrain']
x_test = data['Xtest']
y_test = data['Ytest']