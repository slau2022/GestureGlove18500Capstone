import serial
import keyboard
import numpy as np

arduino = serial.Serial(port='/dev/cu.usbserial-10', baudrate=115200, timeout=.1)
gesture_name = "space"

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
           'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
           'w', 'x', 'y', 'z', 'space', 'random']


going = False
data_point = []

all_data = []

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
np.save("real_data/" + gesture_name + ".npy", all_data)