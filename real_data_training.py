'''
First set of data
SVM Accuracy:  0.8733624454148472
Perceptron Accuracy:  0.759825327510917
KNN Accuracy:  0.8209606986899564
Random forest Accuracy:  0.9679767103347889  
Neural net accuracy:  0.7976710334788938 

real_data_oct26 without magnetometer
SVM Accuracy:  0.7289272030651341
Perceptron Accuracy:  0.5948275862068966
KNN Accuracy:  0.6369731800766284
Random forest Accuracy:  0.9703065134099617 
Neural net accuracy:  0.8065134099616859

real_data_oct26 with magnetometer
SVM Accuracy:  0.7710727969348659
Perceptron Accuracy:  0.64272030651341
KNN Accuracy:  0.6982758620689655
Random forest Accuracy:  0.9645593869731801 
Neural net accuracy:  0.8745210727969349
'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
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

save = False
 
def label_to_num(l):
    if(len(l)==1):
        return ord(l)-96
    elif(l=='space'):
        return 27
    elif(l=='transition'):
        return 28

def num_to_label(n):
    if(n<27):
        return chr(n+97)
    elif(n==27):
        return 'space'
    else: 
        return 'transition'

def parse_data(labels, direc='real_data_oct26/'):  
    all_data = np.empty((0,15))
    #parse label data
    for l in labels: 
        l_file = direc+l+'.npy'
        l_data = np.load(l_file)
        r, c = l_data.shape 
        l_data = np.append(l_data, np.ones((r,1))*label_to_num(l), axis=-1)
     
        all_data = np.vstack((all_data, l_data))


    return all_data

def extract_data(df, no_mag=False):
    if(no_mag): 
        df = df.drop(columns=['s7','s8','s9'])
    headers = df.columns.values
    label = headers[-1]
    features = headers[0:-1]
    # split data
    X, y = df[features], df[label]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)
    #preprocessing data
    scaler = StandardScaler()  
    scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values) 
    X_test = scaler.transform(X_test.values)

    if(save): pickle.dump(scaler,open("scaler_mag.p", "wb"))

    return X_train, y_train, X_test, y_test
 


def train_rf_model(X,y,X_test,y_test):
    #random forest 
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    
    # print(classification_report(y_test,y_pred, labels=[21,22,23,24,27,28], target_names = ['u','v','w','x','space','transition']))
    
    if(save):pickle.dump(rf_model,open("rf_model_mag.p", "wb")) 

    # plot_confusion_matrix(rf_model,X_test,y_test, include_values=False,display_labels=labels)
    # t = rf_model.predict_proba(X[0:1]) 
    # plt.show()

def train_nn_model(X,y, X_test, y_test):

    model = MLPClassifier(solver='lbfgs',alpha=1e-5, 
        hidden_layer_sizes=(10,10,10,10,10),random_state=1, max_iter=1500)
    model.fit(X,y)
    y_pred = model.predict(X_test) 
    print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))
    # plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)
    #validation
    #layer stuff: 
    # 6 later -> 0.776

def train_models(X,y,X_test,y_test): 

    #svm
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=2000)) #SVC(gamma='auto')
    model.fit(X, y)
    y_pred = model.predict(X_test)  
    print("SVM Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)

    #perceptron, we wont use this one probably
    
    model = Perceptron(tol=1e-3, random_state=0)
    model.fit(X,y)
    y_pred = model.predict(X_test)
    print("Perceptron Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)

    #knn 
    model = KNeighborsClassifier(n_neighbors=19).fit(X,y)
    y_pred = model.predict(X_test)  
    print("KNN Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)
    
    #random_forest
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(rf_model,X_test,y_test, include_values=False,display_labels=labels)


    model = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(10,7),random_state=1, max_iter=1000)
    model.fit(X,y)
    y_pred = model.predict(X_test) 
    print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))
    plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)

    # plt.show()

    print(X_test[0])
if __name__ == "__main__":
    df_headers = ['f1','f2','f3','f4','f5','s1',
        's2','s3','s4','s5','s6','s7','s8','s9','label']

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','transition']
    labels_no_tr = letters + ['space']

    data = parse_data(labels,'real_data_oct26/')
    data_no_tr = parse_data(labels_no_tr, 'real_data_oct26/')

    df = pd.DataFrame(data, columns = df_headers)
    df_no_tr = pd.DataFrame(data_no_tr, columns = df_headers)

    all_data = extract_data(df) 
    no_mag_data = extract_data(df, no_mag=True)

    all_data_no_tr = extract_data(df_no_tr) 

    
    print("RF w/ oct26 data")
    train_rf_model(*all_data)
    print("RF w/ oct26 data w/o mag data")
    train_rf_model(*no_mag_data)
    print("RF w/ oct26 data w/o tr label")
    train_rf_model(*all_data_no_tr)
    print("NN w/ oct26 data w/o tr label")
    train_nn_model(*all_data_no_tr)

    # train_rf_model(X_train,y_train,X_test,y_test)
    # train_models(X_train,y_train,X_test,y_test)
 
    #parsing number of letters -> boundaries ... todo
    #try to classify just with random or with space

 

    #todo: confusiong matrix, done
    # neural net: 
    #auto correct: 
    #train without transisiont, done
    


 


 
