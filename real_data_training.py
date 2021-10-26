'''

SVM Accuracy:  0.8733624454148472
Perceptron Accuracy:  0.759825327510917
KNN Accuracy:  0.8209606986899564
Random forest Accuracy:  0.9679767103347889  
Neural net accuracy:  0.7976710334788938
[Finished in 6.0s]

'''
import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import classification_report
 
#nn
from sklearn.neural_network import MLPClassifier
#svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#perceptron
from sklearn.linear_model import Perceptron
#knn
from sklearn.neighbors import KNeighborsClassifier
#random forest
from sklearn.ensemble import RandomForestClassifier

 
def label_to_num(l):
    if(len(l)==1):
        return ord(l)-96
    elif(l=='space'):
        return 27
    elif(l=='random'):
        return 28

def num_to_label(n):
    if(n<27):
        return chr(n+97)
    elif(n==27):
        return 'space'
    else: 
        return 'random'

def parse_data(labels, direc='real_data/'):  
    all_data = np.empty((0,15))
    #parse label data
    for l in labels:
        l_file = direc+l+'.npy'
        l_data = np.load(l_file)
        r, c = l_data.shape 
        l_data = np.append(l_data, np.ones((r,1))*label_to_num(l), axis=-1)
     
        all_data = np.vstack((all_data, l_data))


    return all_data


def train_rf_model(X,y,X_test,y_test):
    #random forest 
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    
    print(classification_report(y_test,y_pred, labels=[27,28], target_names = ['space','random']))
    
    # pickle.dump(rf_model,open("rf_model.p", "wb")) 
 
    # t = rf_model.predict_proba(X[0:1]) 
def train_models(X,y,X_test,y_test): 

    #svm
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=1500)) #SVC(gamma='auto')
    model.fit(X, y)
    y_pred = model.predict(X_test)  
    print("SVM Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=letters)

    #perceptron, we wont use this one probably
    
    model = Perceptron(tol=1e-3, random_state=0)
    model.fit(X,y)
    y_pred = model.predict(X_test)
    print("Perceptron Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=letters)

    #knn 
    model = KNeighborsClassifier(n_neighbors=19).fit(X,y)
    y_pred = model.predict(X_test)  
    print("KNN Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=letters)
    
    #random_forest
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(rf_model,X_test,y_test, include_values=False,display_labels=letters)


    model = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(10,7),random_state=1, max_iter=1000)
    model.fit(X,y)
    y_pred = model.predict(X_test) 
    print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=letters)

    # plt.show()


if __name__ == "__main__":

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','random']

    all_data = parse_data(labels,'real_better_data')
    df = pd.DataFrame(all_data, columns = ['s1','s2','s3','s4','s5','s6',
        's7','s8','s9','s10','s11','s12','s13','s14','label'])

    headers = df.columns.values
    label = headers[-1]
    features = headers[0:-1]
    
    # split data
    from sklearn.model_selection import train_test_split
    X, y = df[features], df[label]
 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)

    #preprocessing data
    from sklearn.preprocessing import StandardScaler   
    scaler = StandardScaler()  
    scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values) 
    X_test = scaler.transform(X_test.values)
    print(X_test[0])
   
    # pickle.dump(scaler,open("scaler.p", "wb"))


    train_rf_model(X_train,y_train,X_test,y_test)

    # train_models(X_train,y_train,X_test,y_test)

    


 


 
