import serial
import time

arduino = serial.Serial(port='/dev/cu.usbmodem101', baudrate=115200, timeout=.1)


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


while True:
    num = input("Enter a number: ")
    exit_program = False
    if (num == "q"):
        exit_program = True
    value = write_read(num)
    if (exit_program):
        break
    print(value)
