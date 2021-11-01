import serial
import keyboard
import numpy as np
import sklearn 
import pickle
from sklearn.preprocessing import StandardScaler  
import time

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

model = pickle.load(open("models/rf_model_oct30_all_data.p", "rb"))
scaler = pickle.load(open("models/scaler_oct30_all_data.p", "rb"))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','random']

going = False
data_point = []

successes = 0
fails = 0

start = time.time()
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
                if (not mag):
                    data_point = data_point[:-3]
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point)
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.25 and pred != 27):
                    print(labels[pred])
                else:
                    print("-")
                successes += 1
            else:
                fails += 1
                
                
            data_point = []
        elif (data == "endbegin"):
            if (len(data_point) == 14):
                if (not mag):
                    data_point = data_point[:-3]
                data_point = np.array(data_point).reshape(1, -1)
                data_point = scaler.transform(data_point)
                #print(data_point)
                probabilities = model.predict_proba(data_point)
                pred = np.argmax(probabilities)
                if (probabilities[0, pred] > 0.25 and pred != 27):
                    print(labels[pred])
                else:
                    print("-")
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

end = time.time()
print("All serial rate:", (successes + fails)/(end - start), "data/s")
print("Success serial rate", successes/(end - start), "data/s")
print("Data fail rate", fails/(successes + fails))