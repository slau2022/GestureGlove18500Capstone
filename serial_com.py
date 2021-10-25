import serial
import keyboard
import numpy as np
import pyttx3
import sklearn 
import pickle
from sklearn.preprocessing import StandardScaler   

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

model = pickle.load(open("rf_model.p", "rb"))
scaler = pickle.load(open("scaler.p", "rb"))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','random']

going = False
data_point = []
while True: 
    if (keyboard.is_pressed("esc")):
        break
    try:
        data = arduino.readline().decode('utf-8').strip()
    except:
        data = ""
    if (going):
        if (data == "end"):
            going = False
            if (len(data_point) == 14):
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point)
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.2 and pred != 27):
                    print(labels[pred])
                else:
                    print("-")
                
                
            data_point = []
        elif (data == "endbegin"):
            if (len(data_point) == 14):
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point)
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.52 and pred != 27):
                    print(labels[pred])
                else:
                    print("-")
            data_point = []
        else:
            try:
                data_point.append(float(data))
            except: 
                pass
    elif (data == "begin"):
        going = True