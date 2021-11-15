
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
#random forest
from sklearn.ensemble import RandomForestClassifier



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

def split_data(df, no_mag=False):
    if(no_mag): 
        df = df.drop(columns=['s7','s8','s9'])
    headers = df.columns.values
    label = headers[-1]
    features = headers[0:-1]
    # split data
    X, y = df[features], df[label]

    return X, y
 
if __name__ == "__main__":
    df_headers = ['f1','f2','f3','f4','f5','s1',
        's2','s3','s4','s5','s6','s7','s8','s9','label']

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','transition']
    labels_no_tr = letters + ['space']

    #testing data
    data = parse_data(labels,'real_data_oct26/')
    df = pd.DataFrame(data, columns = df_headers)
    X, y = split_data(df)

    #trained models
    model = pickle.load(open("models/rf_model_oct30_all_data.p", "rb"))
    scaler = pickle.load(open("models/scaler_oct30_all_data.p", "rb")) 
    #testing
    X = scaler.transform(X)
    y_pred = model.predict(X)
    print("Testing Accuracy: ", metrics.accuracy_score(y,y_pred))
    

    