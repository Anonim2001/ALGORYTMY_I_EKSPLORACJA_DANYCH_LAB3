import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

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


def f1Score(svnPred, param):
    score = f1_score(y_test, svnPred, average='micro')
    print("%s,  F1 score: %s " % (param, score))
    return score


def preparePredicateSVMKernels():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    print('Kernel test')
    bestVal = 0
    bestKernel = ''
    for ker in kernels:
        svcData = svm.SVC(kernel=ker).fit(X_train, y_train)
        score = f1Score(svcData.predict(X_test), ker)
        if (score > bestVal):
            bestVal = score
            bestKernel = ker

    print("Best kernel is %s with score %s" % (bestKernel, bestVal))
    return bestKernel


def preparePredicateSVMC_values(bestKernel):
    c_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8138, 16276]
    print("C param test per kernel: %s" % bestKernel)
    bestVal = 0
    bestC = 1
    for val in c_values:
        svcData = svm.SVC(kernel=bestKernel, C=val).fit(X_train, y_train)
        score = f1Score(svcData.predict(X_test), val)
        if (score > bestVal):
            bestVal = score
            bestC = val

    print("Kernel %s, Best C values is %s with score %s" % (bestKernel, bestC, bestVal))
    return bestC,bestVal

def preparePredicateSVMDegree(bestKernel, c_value):
    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("Degree param test per kernel: %s" % bestKernel)
    bestVal = 0
    bestDegree = 1
    for val in degree:
        svcData = svm.SVC(kernel=bestKernel, C=c_value,degree=val).fit(X_train, y_train)
        score = f1Score(svcData.predict(X_test), val)
        if (score > bestVal):
            bestVal = score
            bestDegree = val

    print("Kernel %s, Best degree values is %s with score %s" % (bestKernel, bestDegree, bestVal))
    return bestDegree,bestVal


if __name__ == '__main__':
    getData()
    kernel = preparePredicateSVMKernels()
    bestKernelPerC = kernel
    c_values = preparePredicateSVMC_values(kernel)
    c_valuesPoly = preparePredicateSVMC_values('poly')
    c_valuesRbf = preparePredicateSVMC_values('rbf')
    if c_values[1] < c_valuesPoly[1]:
        bestKernelPerC = 'poly'
        c_values = c_valuesPoly

    if c_values[1] < c_valuesRbf[1]:
        bestKernelPerC = 'rbf'
        c_values = c_valuesRbf

    print("Best kernel with C values, kernel: %s , C values: %s"% (bestKernelPerC, c_values[0]))
    print("Best F1 score: %s"% (c_values[1]))
    #Wychodzi na to ze najlepszy jest poly z wartoscia C  32
    # zmiana degree
    degree_valuesPoly = preparePredicateSVMDegree(bestKernelPerC,c_values[0])
    print("Best degree: %s , with score: %s "% (degree_valuesPoly[0], degree_valuesPoly[1]))
