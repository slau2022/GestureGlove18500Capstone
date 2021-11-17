import serial
import keyboard
import numpy as np
import sklearn 
import pickle
from sklearn.preprocessing import StandardScaler  
import time
import os
from playsound import playsound

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

model_nn = pickle.load(open("models/nn_model_nov5_all_data.p", "rb"))
model_rf = pickle.load(open("models/rf_model_nov5_all_data.p", "rb"))
model_ruvw = pickle.load(open("models/nn_model_ruvw.p", "rb"))
model_mnste = pickle.load(open("models/nn_model_mnste.p", "rb"))
scaler = pickle.load(open("models/scaler_nov5_all_data.p", "rb"))
scaler_ruvw = pickle.load(open("models/scaler_ruvw.p", "rb"))
scaler_mnste = pickle.load(open("models/scaler_mnste.p", "rb"))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','transition']

ruvw = ["r", "u", "v", "w"]
mnste = ["m", "n", "s", "t", "e"]

going = False
data_point = []

vector_len = 14
window_size = 3

successes = 0

start = time.time()
'''while True: 
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
        going = True'''

data_point = []
arduino.write(bytes("go", "utf-8"))
while (True):
    if (keyboard.is_pressed("esc")):
        break
    try:
        data = arduino.readline().decode('utf-8').strip()
    except:
        data = ""
    try:
        data = data.strip().split(",")
        if (len(data) == vector_len):
            for i in range(vector_len):
                    data[i] = float(data[i])
            data_point.extend(data)
        else: 
            arduino.write(bytes("go", "utf-8"))
            continue
        if (len(data_point) == vector_len*window_size):
            data_point_or = np.array(data_point).reshape(1, -1)
            data_point = scaler.transform(data_point_or)
            probabilities_rf = model_rf.predict_proba(data_point)
            pred_rf = np.argmax(probabilities_rf)
            if (probabilities_rf[0, pred_rf] > 0.25 and pred_rf != 27 and pred_rf != 26):
                if (labels[pred_rf] in {"r", "u", "v", "w"}):
                    pred = int(model_ruvw.predict(scaler_ruvw.transform(data_point_or))[0])
                    #playsound("audio_files/" + labels[pred_rf] + ".mp3")
                    print(labels[pred-1].upper(), labels[pred_rf].upper())
                elif (labels[pred_rf] in {"m", "n", "s", "t", "e"}):
                    pred = int(model_mnste.predict(scaler_mnste.transform(data_point_or))[0])
                    print(labels[pred-1].upper(), labels[pred_rf].upper())
                else:
                    print(labels[pred_rf].upper())
            else:
                print("-")
            data_point = []
            successes += 1
            arduino.write(bytes("go", "utf-8"))
    except:
        print("uh oh")
        data_point = []

end = time.time()

pred_rate = successes/(end - start)
print("Prediction rate:", pred_rate)

data_file = "serial_com_data.txt"
if (not os.path.exists(data_file)):
    f = open(data_file, "w")
    f.write("Prediction rate: " + str(pred_rate) + "predictions/sec\n\n")
    f.close()
else:
    f = open(data_file, "a")
    f.write("Prediction rate: " + str(pred_rate) + "predictions/sec\n\n")
    f.close()