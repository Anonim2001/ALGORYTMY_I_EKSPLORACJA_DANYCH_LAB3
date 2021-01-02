import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

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


def generateConfusionMetrix(prediction, name):
    cm = confusion_matrix(y_test, prediction)
    df_cm = pd.DataFrame(cm, range(12), range(12))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')

    fig = plt.gcf()
    plt.show()
    fig.savefig("%s_confusion_matrix.png" % name)

def rocauc(pred):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y = lb.transform(y_test)
    pred = lb.transform(pred)
    return roc_auc_score(y, pred, average='macro')

if __name__ == '__main__':
    getData()
    svnPred = preparePrediction('svn')
    kncPred = preparePrediction('knc')
    dtcPred = preparePrediction('dtc')
    rfcPred = preparePrediction('rfc')
    generateConfusionMetrix(svnPred,'svn')
    generateConfusionMetrix(kncPred, 'knc')
    generateConfusionMetrix(dtcPred,'dtc')
    generateConfusionMetrix(svnPred, 'rfc')
    print("svn ACC: %s"% accuracy_score(y_test, svnPred))
    print("knc ACC: %s"% accuracy_score(y_test, kncPred))
    print("dtc ACC: %s"% accuracy_score(y_test, dtcPred))
    print("rfc ACC: %s"% accuracy_score(y_test, rfcPred))

    print("svn recall: %s"% recall_score(y_test, svnPred, average='micro'))
    print("knc recall: %s"% recall_score(y_test, kncPred, average='micro'))
    print("dtc recall: %s"% recall_score(y_test, dtcPred, average='micro'))
    print("rfc recall: %s"% recall_score(y_test, rfcPred, average='micro'))

    print("svn f1: %s"% f1_score(y_test, svnPred, average='micro'))
    print("knc f1: %s"% f1_score(y_test, kncPred, average='micro'))
    print("dtc f1: %s"% f1_score(y_test, dtcPred, average='micro'))
    print("rfc f1: %s"% f1_score(y_test, rfcPred, average='micro'))

    print("svn auc: %s"% rocauc(svnPred))
    print("knc auc: %s"% rocauc(kncPred))
    print("dtc auc: %s"% rocauc(dtcPred))
    print("rfc auc: %s"% rocauc(rfcPred))