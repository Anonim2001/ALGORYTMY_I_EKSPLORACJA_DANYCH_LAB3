import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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


def prepareClassifiers(type):
    if type == 'svn':
        svcData = svm.SVC().fit(X_train, y_train)
        return svcData
    elif type == 'knc':
        kncData = KNeighborsClassifier().fit(X_train, y_train)
        return kncData
    elif type == 'dtc':
        dtcData = DecisionTreeClassifier().fit(X_train, y_train)
        return dtcData
    elif type == 'rfc':
        rfcData = RandomForestClassifier().fit(X_train, y_train)
        return rfcData


def calAverage(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg


if __name__ == '__main__':
    getData()
    svnClassifiers = prepareClassifiers('svn')
    kncClassifiers = prepareClassifiers('knc')
    dtcClassifiers = prepareClassifiers('dtc')
    rfcClassifiers = prepareClassifiers('rfc')
    cvsSvn = cross_val_score(svnClassifiers, X_train, y_train, cv=5)
    cvsKnc = cross_val_score(kncClassifiers, X_train, y_train, cv=5)
    cvsDtc = cross_val_score(dtcClassifiers, X_train, y_train, cv=5)
    cvsRfc = cross_val_score(rfcClassifiers, X_train, y_train, cv=5)
    print('svn')
    print(cvsSvn)
    print("svn avg %f, std %f" % (calAverage(cvsSvn), np.std(cvsSvn)))
    print('knc')
    print(cvsKnc)
    print("knc avg %f, std %f" % (calAverage(cvsKnc), np.std(cvsKnc)))
    print('dtc')
    print(cvsDtc)
    print("dtc avg %f, std %f" % (calAverage(cvsDtc), np.std(cvsDtc)))
    print('rfc')
    print(cvsRfc)
    print("rfc avg %f, std %f" % (calAverage(cvsRfc), np.std(cvsRfc)))
