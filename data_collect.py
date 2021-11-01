from numpy.core.numeric import full
import serial
import keyboard
import numpy as np
import time
import os

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

directory = "window3/"
if (not os.path.isdir(directory)):
    os.mkdir(directory)

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','random']
gesture_name = "a"
gesture_ind = 0
vector_len = 14
window_size = 3


going = False
data_point = []
full_point = []

all_data = []

print("COLLECTING FOR LABEL", gesture_name)
print("Go")

while True: 
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
    except:
        data = ""
    if (going):
        if (data == "end"):
            going = False
            if (len(data_point) == vector_len):
                full_point.extend(data_point)
                if (len(full_point) == vector_len*window_size):
                    if (keyboard.is_pressed("space")):
                        all_data.append(full_point)
                        print(full_point)
                    full_point = []
                data_point = []
                    
        elif (data == "endbegin"):
            going = True
            if (len(data_point) == vector_len):
                full_point.extend(data_point)
                if (len(full_point) == vector_len*window_size):
                    if (keyboard.is_pressed("space")):
                        all_data.append(full_point)
                        print(full_point)
                    full_point = []
                data_point = []
        else:
            going = True
            try:
                data_point.append(float(data))
            except: 
                pass
    if (data == "begin"):
        print("hi")
        going = True
    
    if (going == False and data != "begin"):
        print('uh oh')

all_data = np.array(all_data)
np.save(directory + gesture_name + ".npy", all_data)