import csv
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import numpy as np
import random 
from sklearn.metrics import ConfusionMatrixDisplay

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['transition']

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

def labels_to_nums(labels):
    result = list()
    for l in labels:
        result.append(label_to_num(l))
    return result 

def nums_to_labels(nums):
    result = list()
    for n in nums:
        result.append(num_to_label(n))
    return result



 

def data_reconstruction():
    file = open("accuracies.csv")
    csvreader = csv.reader(file)

    y_true = list()
    y_pred = list()
    for row in csvreader:
        letter = row[0]
        data = row[1].split(',')
        # print(data[0]) 
        correct =  int(data[1])
        total =  int(data[2].strip(']')) 
        y_true = y_true + [letter]*total
        #[random.choice(labels)]
        y_pred = y_pred + [letter]*correct + ['transition']*(total-correct)

    file.close() 

  
    ConfusionMatrixDisplay.from_predictions( y_true, y_pred, labels=labels, include_values=False, cmap=plt.cm.Blues, normalize='true')
    #use true
    # cmap=plt.cm.Blues

    plt.show()
data_reconstruction()