import serial
import keyboard
import numpy as np
import sklearn 
import pickle
from sklearn.preprocessing import StandardScaler  
import time
import os
from playsound import playsound
import warnings


arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

model_rf = pickle.load(open("models/rf_model_dec11_2.p", "rb"))
scaler = pickle.load(open("models/ct_dec7_5050.p", "rb"))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','transition']

warnings.filterwarnings("ignore")

ruvw = ["r", "u", "v", "w"]
mnste = ["m", "n", "s", "t", "e"]

going = False
data_point = []

vector_len = 17
window_size = 3

successes = 0
num_pred = 0

start = time.time()

data_point = []
prev = ""
prev_outputted = ""
broken = False
time.sleep(2)
while (True):
    arduino.write(bytes("go", "utf-8"))
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
                if ((not broken) and i < 5 and float(data[i]) == 0.0):
                    print('WARNING: Sensor', i, "is loose", end="... ")
                    broken = True
                data[i] = float(data[i])
                if (broken and (0.0 not in data[:5])):
                    print("fixed!")
                    broken = False
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
                if (prev == "" or prev[-1] != labels[pred_rf]):
                    prev = labels[pred_rf]
                else:
                    prev += labels[pred_rf]
            else:
                print("-")
                prev = ""
            data_point = []
            if (len(prev) == 10):
                if (prev_outputted != prev[-1]): 
                    print(prev[-1])
                    playsound("audio_files/" + prev[-1] + ".mp3")
                    prev_outputted = prev[-1]
                else:
                    prev_outputted = ""
                successes += 1
                prev = ""
            num_pred += 1
    except:
        data_point = []
    arduino.write(bytes("go", "utf-8"))

end = time.time()

pred_rate = successes/(end - start)
pred_time = (end - start)/num_pred
print("Prediction rate:", pred_rate)
print("Time per prediction:", pred_time)

data_file = "serial_com_data.txt"
if (not os.path.exists(data_file)):
    f = open(data_file, "w")
    f.write("Prediction rate: " + str(pred_rate) + "predictions/sec\n \
             Time per prediction: " + str(pred_time) + "sec/prediction\n\n")
    f.close()
else:
    f = open(data_file, "a")
    f.write("Prediction rate: " + str(pred_rate) + "predictions/sec\n \
            Time per prediction: " + str(pred_time) + "sec/prediction\n\n")
    f.close()