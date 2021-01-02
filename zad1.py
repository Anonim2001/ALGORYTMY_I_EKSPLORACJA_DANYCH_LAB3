import numpy as np

global X_train
global y_train
global X_test
global y_test


def getData():
    global X_train
    global y_train
    global X_test
    global y_test
    X_train = np.loadtxt('X_train.txt', delimiter=' ')
    y_train = np.loadtxt('y_train.txt')

    X_test = np.loadtxt('X_test.txt', delimiter=' ')
    y_test = np.loadtxt('y_test.txt')


if __name__ == '__main__':
    getData()
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
