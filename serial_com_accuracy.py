import serial
import keyboard
import numpy as np
import sklearn 
import pickle
from sklearn.preprocessing import StandardScaler  
import time
import csv

'''
COMMONLY CONFUSED LETTERS:
c + o
d + t + x
n + a
r + u + v
t + m + n
u + w + k
x (a big problem)
z + l
'''

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

mag = True

model = pickle.load(open("models/rf_model_oct26_all_data.p", "rb"))
scaler = pickle.load(open("models/scaler_oct26_all_data.p", "rb"))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','random']

going = False
data_point = []

successes = 0
fails = 0

start = time.time()
label_index = 0
correct = 0
total = 0
accuracies = dict()

while (True and label_index < len(labels)): 
    curr_label = labels[label_index]
      
    if (keyboard.is_pressed("esc")):
        break
    elif(keyboard.is_pressed("enter") and total!=0):
        print(correct,total)
        accuracies[curr_label] = [ correct/(total+1),correct, total]
        print(f'Accuracy for {curr_label} is {correct/total*100:.2f}%')
        label_index+=1
        correct = 0
        total = 0
 
        print("Caclulating accuracy for: ", labels[label_index])   
       

    try:
        data = arduino.readline().decode('utf-8').strip()
    except:
        data = ""

    if (going):
        if (data == "end"):
            going = False
            if (keyboard.is_pressed("space") and len(data_point) == 14):
                 
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point)
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.25 and pred != 27):
                    print(labels[pred])
                    if(labels[pred] == curr_label):
                        correct+=1
                else:
                    print("-")
                total+=1
                successes += 1
            else:
                fails += 1
                
                
            data_point = []
        elif (data == "endbegin"):
            if (keyboard.is_pressed("space") and len(data_point) == 14):
                 
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point) 
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.25 and pred != 27):
                    print(labels[pred])
                    if(labels[pred] == curr_label):
                        correct+=1
                else:
                    print("-")
                total+=1
                successes += 1
            else:
                fails += 1
            data_point = []
        else:
            try:
                data_point.append(float(data))
            except: 
                pass
    elif (data == "begin"):
        going = True

w = csv.writer(open("accuracies.csv", "w")) 
for key, val in accuracies.items(): 
    w.writerow([key, val])

end = time.time()
print("All serial rate:", (successes + fails)/(end - start), "data/s")
print("Success serial rate", successes/(end - start), "data/s")
print("Data fail rate", fails/(successes + fails))