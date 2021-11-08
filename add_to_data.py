import serial
import keyboard
import numpy as np
import time

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = letters + ['space','transition']
gesture_name = "a"
gesture_ind = 0
directory = "window3/"

vector_len = 14
window_size = 3

'''going = False
data_point = []

all_data = list(np.load(directory + gesture_name + ".npy"))

print("COLLECTING FOR LABEL", gesture_name)
print("Press 'space' to collect data and 'esc' to go to next letter")

while (True):
    if (keyboard.is_pressed("esc")):
        if (gesture_ind == 27):
            break
        else:
            all_data = np.array(all_data)
            np.save(directory + gesture_name + ".npy", all_data)
            gesture_ind += 1
            gesture_name = labels[gesture_ind]
            all_data = list(np.load(directory + gesture_name + ".npy"))
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
            if (keyboard.is_pressed("space") and len(data_point) == 14):
                print(data_point)
                all_data.append(data_point)

            data_point = []
        elif (data == "endbegin"):
            if (keyboard.is_pressed("space") and len(data_point) == 14):
                print(data_point)
                all_data.append(data_point)
            data_point = []
        else:
            try:
                data_point.append(float(data))
            except: 
                pass
    elif (data == "begin"):
        going = True

all_data = np.array(all_data)
np.save(directory + gesture_name + ".npy", all_data)'''

data_point = []
all_data = list(np.load(directory + gesture_name + ".npy"))

print("COLLECTING FOR LABEL", gesture_name)
print("Go")

while (True):
    if (keyboard.is_pressed("esc")):
        if (gesture_ind == 27):
            break
        else:
            all_data = np.array(all_data)
            np.save(directory + gesture_name + ".npy", all_data)
            gesture_ind += 1
            gesture_name = labels[gesture_ind]
            try:
                all_data = list(np.load(directory + gesture_name + ".npy"))
            except:
                all_data = []
            print("COLLECTING FOR LABEL", gesture_name)
            time.sleep(3)
            print("Go")
    try:
        data = arduino.readline().decode('utf-8').strip()
    except:
        data = ""
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

all_data = np.array(all_data)
np.save(directory + gesture_name + ".npy", all_data)