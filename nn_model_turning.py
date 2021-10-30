 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
 

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
#hyperparameter tuing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold 
from scipy.stats import loguniform 
from scipy.stats import randint as sp_randint

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

#extract training and testing data from a df
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
 


def train_nn_model(X,y,X_test, y_test):

    model = MLPClassifier(solver='lbfgs',alpha=1e-5, 
        hidden_layer_sizes=(50,25,15),random_state=1, max_iter=1500)
    model.fit(X,y)
    y_pred = model.predict(X_test) 
    print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))
    # plot_confusion_matrix(model,X_test,y_test, include_values=False,display_labels=labels)
    #validation
    #layer stuff: 
    # 6 layer -> 0.776
    # 5 layer - > 0.77, 4 -> 0.69, 3->0.8684
def tune_nn_model(X,y, X_test, y_test, name=None):

    parameter_space = {
    'hidden_layer_sizes': 
    [(sp_randint.rvs(1,150,1),
        sp_randint.rvs(1,150,1),), (sp_randint.rvs(1,150,1),)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': loguniform(1e-5,1e-1) ,
    'learning_rate': ['constant','adaptive']}

    model = MLPClassifier(max_iter=1000)

    cvFold = RepeatedKFold(n_splits=15, n_repeats=3)
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    cv=cvFold, param_distributions=parameter_space) #use default?
    #scoring="neg_mean_squared_error"
    searchResults = randomSearch.fit(X, y)

    # extract the best model and evaluate it
    print("[INFO] evaluating...")
    best_model = searchResults.best_estimator_
    #Save best model
    pickle.dump(best_model,open(name+".p", "wb"))
    print(best_model) 

    #save resutls
    df = pd.DataFrame(searchResults.cv_results_)  
    if(name==None): 
        df.to_csv()
    else:
        df.to_csv(name + ".csv")

    print("R2: {:.5f}".format(best_model.score(X_test, y_test)))



 
if __name__ == "__main__":
    df_headers = ['f1','f2','f3','f4','f5','s1',
        's2','s3','s4','s5','s6','s7','s8','s9','label']

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','transition']
    labels_no_tr = letters + ['space']
 
    data = parse_data(labels,'real_data_oct26/')
    data_no_tr = parse_data(labels_no_tr, 'real_data_oct26/')
    data_letters = parse_data(letters,'real_data_oct26/')

    df = pd.DataFrame(data, columns = df_headers)
    df_no_tr = pd.DataFrame(data_no_tr, columns = df_headers)
    df_letters = pd.DataFrame(data_letters, columns = df_headers)

    all_data = extract_data(df) 
    no_mag_data = extract_data(df, no_mag=True)

    data_no_tr = extract_data(df_no_tr) 
    data_letters = extract_data(df_letters) 

    
    print("NN w/ oct26 data w tr label") 
    tune_nn_model(*all_data, "nn_tuning_all_data")
 
    print("NN w/ oct26 data w/o tr label") 
    tune_nn_model(*data_no_tr, "nn_tuning_no_tr")

    print("NN w/ oct26 data w/ only letters")
    tune_nn_model(*data_letters, "nn_tuning_letters")







 


 


 
