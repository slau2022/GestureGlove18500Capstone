from numpy.core.numeric import full
import serial
import keyboard
import numpy as np
import time
import os

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

directory = "dec7_stephanie/"
if (not os.path.isdir(directory)):
    os.mkdir(directory)

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','transition']
gesture_name = "a"
gesture_ind = 0
vector_len = 17
window_size = 3



data_point = []
all_data = []

print("COLLECTING FOR LABEL", gesture_name)
print("Go")

while (True):
    arduino.write(bytes("go", "utf-8"))
    if (keyboard.is_pressed("esc")):
        if (gesture_ind == 27):
            break
        else:
            all_data = np.array(all_data)
            np.save(directory + gesture_name + ".npy", all_data)
            gesture_ind += 1
            gesture_name = labels[gesture_ind]
            all_data = []
            print("COLLECTING FOR LABEL", gesture_name)
            time.sleep(3)
            print("Go")
    try:
        data = arduino.readline().decode('utf-8').strip()
        data = data.strip().split(",")
        if (len(data) == vector_len):
            for i in range(vector_len):
                data[i] = float(data[i])
            data_point.extend(data)
        if (len(data_point) == vector_len*window_size):
            if (keyboard.is_pressed("space")):
                print(data_point)
                all_data.append(data_point)
            data_point = []
    except:
        data = ""
    

all_data = np.array(all_data)
np.save(directory + gesture_name + ".npy", all_data)