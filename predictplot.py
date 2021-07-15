

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


# Create function to return

def predictplot(Model, xdata, ydata):
    
    y_pred = Model.predict(xdata)

    logit_roc_auc = roc_auc_score(ydata, Model.predict(xdata))
    fpr, tpr, thresholds = roc_curve(ydata, Model.predict_proba(xdata)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label= type(Model).__name__ + '(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print(classification_report(ydata, y_pred))
    return