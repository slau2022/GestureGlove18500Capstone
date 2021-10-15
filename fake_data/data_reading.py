## sample code for reading in data ##




'''
SVM Accuracy:  0.7361538461538462
Perceptron Accuracy:  0.6369230769230769
KNN Accuracy:  0.7438461538461538
Random forest Accuracy:  0.8176923076923077  
Neural net accuracy:  0.7307692307692307
'''

import pandas as pd  
 

train_df = pd.read_csv('simulated_train_data.csv',delimiter=' ') 
test_df = pd.read_csv('simulated_test_data.csv',delimiter=' ')
headers = train_df.columns.values
label = headers[0]
features = headers[1:]


## helper methods ##
def char_to_num(l):
    return ord(l)-96

def num_to_char(n):
    return chr(n+97)
 