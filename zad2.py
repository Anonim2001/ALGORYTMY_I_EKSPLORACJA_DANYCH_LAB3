import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


def preparePrediction(type):
    if type == 'svn':
        svcData = svm.SVC().fit(X_train, y_train)
        return svcData.predict(X_test)
    elif type == 'knc':
        kncData = KNeighborsClassifier().fit(X_train, y_train)
        return kncData.predict(X_test)
    elif type == 'dtc':
        dtcData = DecisionTreeClassifier().fit(X_train, y_train)
        return dtcData.predict(X_test)
    elif type == 'rfc':
        rfcData = RandomForestClassifier().fit(X_train, y_train)
        return rfcData.predict(X_test)


if __name__ == '__main__':
    getData()
    print('svn')
    print(preparePrediction('svn'))
    print('knc')
    print(preparePrediction('knc'))
    print('dtc')
    print(preparePrediction('dtc'))
    print('rfc')
    print(preparePrediction('rfc'))
