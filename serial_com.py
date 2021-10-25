import serial
import keyboard
import numpy as np
import pyttsx3

arduino = serial.Serial(port='/dev/cu.usbserial-10', baudrate=115200, timeout=.1)

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
                print(data_point)
                
                # MAKE MODEL PREDICTION + print
            data_point = []
        elif (data == "endbegin"):
            if (len(data_point) == 14):
                print(data_point)
                # MAKE MODEL PREDICTION + print
            data_point = []
        else:
            try:
                data_point.append(float(data))
            except: 
                pass
    elif (data == "begin"):
        going = True