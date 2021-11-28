 
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

#hyperparameter tuing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedKFold 
from scipy.stats import loguniform 
from scipy.stats import randint as sp_randint

save = False

from sklearn.exceptions import ConvergenceWarning 
ConvergenceWarning('ignore')
from warnings import filterwarnings
filterwarnings('ignore')

def label_to_num(l):
    if(len(l)==1):
        return ord(l)-96
    elif(l=='space'):
        return 27
    elif(l=='transition'):
        return 28

def num_to_label(n):
    if(n<27):
        return chr(n+96)
    elif(n==27):
        return 'space'
    else: 
        return 'transition'


#go thru each sample, if value in sample > 500 (outlier) 
def remove_outliers(data):
    flex_index = [0,1,2,3,4,14,15,16,17,18,28,29,30,31,32]
    if(len(data.shape)==1): return data
    r,c = data.shape
    #data[0] = row; data[:,0] = column 
    # print(type(data))
    to_del = list()
    for i in range(0,r):
        row = data[i] 
        for j,val in enumerate(row):
            
            if(val<=-40 and j in flex_index): 
                to_del.append(i)
                break 

            if(val>=500): 
                to_del.append(i)
                break  
    #(3788, 43) (15934, 43) (4746, 43)
    #(3779, 43) (15934, 43) (4736, 43)
    new_data = np.delete(data, to_del, axis=0) #delete row
 

    return new_data


def parse_data(labels, direc='real_data_nov5_window3/'):
    data_len = 14*3+1
    all_data = np.empty((0,data_len))
    #parse label data
    for l in labels: 
        l_file = direc+l+'.npy'
        l_data = np.load(l_file)
        l_data = remove_outliers(l_data)

        r, c = l_data.shape  

        l_data = np.append(l_data, np.ones((r,1))*label_to_num(l), axis=-1)
     
        all_data = np.vstack((all_data, l_data))


    return all_data


def extract_data(df, no_mag=False, name = "scaler_xxx.p"):
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

    if(save): pickle.dump(scaler,open("models/"+name, "wb"))

    return X_train, y_train, X_test, y_test
 
def train_rf_model(X,y,X_test,y_test, labels, name = "rf_model_xxx.p"):
    #random forest 
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))
    
    # print(classification_report(y_test,y_pred, labels=[21,22,23,24,27,28], target_names = ['u','v','w','x','space','transition']))
    
    if(save):pickle.dump(rf_model,open("models/"+name, "wb")) 

    plot_confusion_matrix(rf_model,X_test,y_test, include_values=False,display_labels=labels)
    plt.show()
    # t = rf_model.predict_proba(X[0:1]) 

def train_nn_model(X,y, X_test, y_test, labels,name = "nn_model_xxx.p"):

    # model = MLPClassifier(solver='lbfgs',alpha=0.0001435385074058134, 
    #     hidden_layer_sizes=(56, 113), activation='logistic', max_iter=1000)
    nn_model = MLPClassifier(alpha=0.00037009903911428525, hidden_layer_sizes=(134, 25), learning_rate='adaptive', max_iter=1500, solver='lbfgs')
    nn_model.fit(X,y)
    y_pred = nn_model.predict(X_test) 
    print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))

    if(save): pickle.dump(nn_model, open("models/"+name, "wb"))
    # plot_confusion_matrix(nn_model,X_test,y_test, include_values=False,display_labels=labels)
    # plt.show()  
    '''
    NN w/ oct26 data w tr label 
    MLPClassifier(activation='logistic', alpha=0.0001435385074058134,
                  hidden_layer_sizes=(56, 113), max_iter=1000)
    R2: 0.95115
    '''

def make_acc_barplot(X,y,X_test,y_test): 
    models = ['SVM','Perceptron','KNN','Random Forest', 'NN (Validated)']
    accs = list()
    #svm
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=2000)) #SVC(gamma='auto')
    model.fit(X, y)
    y_pred = model.predict(X_test)  
    accs.append(metrics.accuracy_score(y_test,y_pred))
 

    #perceptron, we wont use this one probably
    
    model = Perceptron(tol=1e-3, random_state=0)
    model.fit(X,y)
    y_pred = model.predict(X_test)
    accs.append( metrics.accuracy_score(y_test,y_pred))
 
    #knn 
    model = KNeighborsClassifier(n_neighbors=19).fit(X,y)
    y_pred = model.predict(X_test)  
    accs.append(metrics.accuracy_score(y_test,y_pred))
 
    #random_forest
    rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = rf_model.predict(X_test) 
    accs.append(metrics.accuracy_score(y_test,y_pred))
 

    # model = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(10,7),random_state=1, max_iter=1000)  
    model = MLPClassifier(alpha=0.00037009903911428525, hidden_layer_sizes=(134, 25), learning_rate='adaptive', max_iter=1500, solver='lbfgs')
    model.fit(X,y)
    y_pred = model.predict(X_test) 
    accs.append(metrics.accuracy_score(y_test,y_pred)) 

    plt.bar(models, accs, width=0.7)
    plt.xlabel("Models")
    plt.ylabel("Accuracies")
    plt.title("Models' accuracies")
    plt.show()
def tune_rf_model(X,y,X_test,y_test, name='rf_model'): 
    parameter_space = {
    'n_estimators': [sp_randint.rvs(50,150,1)] ,
    'min_samples_split': [sp_randint.rvs(1,150,1)],
    'min_samples_leaf': [sp_randint.rvs(1,100,1)] }

    model = RandomForestClassifier()

    cvFold = RepeatedKFold(n_splits=5, n_repeats=30)
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    cv=cvFold, param_distributions=parameter_space) #use default?
    gridSearch = GridSearchCV(estimator=model, n_jobs=-1, param_grid=parameter_space, cv=20, verbose=2)
    #scoring="neg_mean_squared_error"
    # searchResults = randomSearch.fit(X, y)
    searchResults = gridSearch.fit(X,y)

    # extract the best model and evaluate it 
    best_model = searchResults.best_estimator_
    #Save best model 
    pickle.dump(best_model,open(name+".p", "wb"))
    print(best_model) 
    # model.fit(X,y)
    y_pred = best_model.predict(X_test)

    #save resutls
    df = pd.DataFrame(searchResults.cv_results_)  
    if(name==None): 
        df.to_csv()
    else:
        df.to_csv(name + ".csv")

    #metrics.accuracy_score
    print("Accuracy: {:.5f}".format( metrics.accuracy_score(y_test, y_pred))) 
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
def make_hyperparameter_graph(X,y, X_test,y_test):
    #     parameter_space = {
    # 'n_estimators': [sp_randint.rvs(50,150,1)] ,
    # 'min_samples_split': [sp_randint.rvs(1,150,1)],
    # 'min_samples_leaf': [sp_randint.rvs(1,100,1)] }
    #random forest: n_estimators: number of trees
    limit = 300
    n_estimators= range(2,limit)
    acc1 = list()
    for n in range(2,limit): 
        print("n_estimators = ", n)
        rf_model = RandomForestClassifier(n_estimators=n).fit(X,y)
        y_pred = rf_model.predict(X_test)  
        acc1.append(metrics.accuracy_score(y_test,y_pred)) 


    plt.figure()
    plt.plot(n_estimators, acc1)
    plt.xlabel("n_estimators (# of trees)")
    plt.ylabel("accuracy") 
    plt.savefig("graphs/n_estimators vs accuracy")

    min_samples_split= range(2,limit)
    acc1 = list()
    for n in range(2,limit): 

        print("min_samples_split = ", n)
        rf_model = RandomForestClassifier(min_samples_split=n).fit(X,y)
        y_pred = rf_model.predict(X_test) 
        acc1.append(metrics.accuracy_score(y_test,y_pred)) 

    plt.figure()
    plt.plot(min_samples_split, acc1)
    plt.xlabel("min_samples_split")
    plt.ylabel("accuracy") 
    plt.savefig("graphs/min_samples_split vs accuracy")

    min_samples_leaf= range(2,limit)
    acc1 = list()
    for n in range(2,limit):

        print("min_samples_leaf = ", n) 
        rf_model = RandomForestClassifier(min_samples_leaf=n).fit(X,y)
        y_pred = rf_model.predict(X_test) 
        acc1.append(metrics.accuracy_score(y_test,y_pred)) 

    plt.figure()
    plt.plot(min_samples_leaf, acc1)
    plt.xlabel("min_samples_leaf")
    plt.ylabel("accuracy") 
    plt.savefig("graphs/min_samples_leaf vs accuracy")

def make_headers(headers):
    new_header = list()
    count=1
    window_size=3
    while(count<window_size):

        for h in headers:
            temp = h+"_"+str(count)
            new_header.append(temp)
        count+=1
    return headers+new_header


if __name__ == "__main__":
    save=False
    data_path = 'all data'
    df_headers = ['f1','f2','f3','f4','f5','s1',
        's2','s3','s4','s5','s6','s7','s8','s9'] 
    df_headers = make_headers(df_headers)
    
    df_headers = df_headers + ['label']

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','transition']
    labels_no_tr = letters + ['space']

    paths = ['real_data_nov5_window3/','rachel_roommates/','data_sz_friends/']
    data = np.empty( (0, 43)) 
    data_no_tr = np.empty( (0,43))
    for path in paths:
        temp1 = parse_data(labels, path) 
        temp2 = parse_data(labels_no_tr,path)
        data = np.concatenate( (data, temp1), axis=0) 
        data_no_tr = np.concatenate( (data,temp2), axis=0)
 
    df = pd.DataFrame(data, columns = df_headers)
    df_no_tr = pd.DataFrame(data_no_tr, columns = df_headers) 

    save=True
    all_data = extract_data(df, name = "scaler_nov22_all_data.p") 
    # no_mag_data = extract_data(df, no_mag=True)
    # all_data_no_tr = extract_data(df_no_tr, name="scaler_nov22_no_tr.p") 
 
 
    print("RF w/ " + data_path)
    train_rf_model(*all_data,labels = labels,  name = "rf_model_nov22_all_data.p")
  

 


 
