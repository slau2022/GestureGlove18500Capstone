from numpy.random import default_rng
import numpy as np
import csv
rng=default_rng()
'''
IMU data: accelerometer, gyroscope, magnetometer
-accelerometer shouldnt matter for still gesture: randomly generated between -1 to 1
-gyroscope, probably shouldn’t matter? Randomly generated between -5 to 5
-magnetometer: directions facing,

Thumbs up: W~S
85~97 degrees for each finger besides thumb, thumb = 0~10 degrees
Thumbs down: E~NE
Similar as thumbs up
Peace sign: N~NNW
1: 40-60; 2,3: 0-15; 4,5:~90
Love sign: N~NNW
1: 0-10; 2: 0-15; 3,4: 85~97; 5: 10-30
Stop sign: N~NNW
1-5: 0-15
'''
#normal distribution for flex sensors
#random for directions

#a data vector should have 5+9=14 numbers

# Five gestures
'''
THUMBS_UP = 1
THUMBS_DOWN = 2
PEACE = 3
LOVE = 4 
STOP = 5
gesture = [THUMBS_UP, THUMBS_DOWN,PEACE,LOVE,STOP]
data_range = dict()
data_range[THUMBS_UP] = [5,5, 91,6, 91,6, 91,6, 91,6, -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 16,35, 11,35, 0,4]
data_range[THUMBS_DOWN] = [5,5, 91,6, 91,6, 91,6, 91,6, -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -0.3,11, 5.5,17, 4,8]
data_range[PEACE] = [50,10, 7,7, 7,7, 90,5, 90,5,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range[LOVE] = [5,5, 7.5,7.5, 90,6, 90,6, 20,10,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range[STOP] = [7.5,7.5, 7.5,7.5, 7.5,7.5, 7.5,7.5, 7.5,7.5,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
''' 

# Letters
# Flex sensors: normal variation (mean, var)
# IMUs: random number within a range (min, max)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = [ord(l)-96 for l in letters]
data_range = dict()
data_range['a'] = [ 0,5, 91,6, 91,6, 91,6, 91,6,    
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4] #north
data_range['b'] = [50,10, 7.5,7.5, 7.5,7.5, 7.5,7.5, 7.5,7.5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4 ]
data_range['c'] = [30,5, 70,10, 70,10, 70,10, 70,10,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['d'] = [50,10, 7,7, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range['e'] = [90,5, 91,6, 91,6, 91,6, 91,6, 
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range['f'] = [40,5, 120,10,  7.5,7.5, 7.5,7.5, 7.5,7.5, 
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4] 
data_range['g'] = [0,5, 7,7, 90,5, 90,5, 90,5,
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4] #west
data_range['h'] = [0,5, 7,7,  7,7, 90,5, 90,5,
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['i'] = [30,5, 90,5, 90,5, 90,5, 0,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['j'] = [30,5, 90,5, 90,5, 90,5, 0,5,  
-1,1, -1,1, -1,1, -30,-5, 100,120, -5,5, 14,20, 8,14, 0,4] #moving
data_range['k'] = [30,10, 7,7, 7,7, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['l'] = [-30,10, 7,7, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['m'] = [65,5, 91,6, 91,6, 91,6, 110,6,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['n'] = [45,5, 91,6, 91,6, 91,6, 110,6,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4] 
data_range['o'] = [45,5, 77,7, 77,7, 77,7, 77,7,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['p'] = [30,10, 7,7, 7,7, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 28,32, 28,32, 3,5] #south
data_range['q'] = [0,5, 7,7, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 28,32, 28,32, 3,5]
data_range['r'] = [15,5, 13,4, 2,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['s'] = [30,5, 90,5, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['t'] = [17,5, 65,7, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['u'] = [50,10, 7,7, 7,7, 90,5, 90,5,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['v'] = [50,10, 2,4, 2,4, 90,5, 90,5,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['w'] = [50,10, 7,7, 7,7, 7,7, 90,5,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['x'] = [80,10, 70,5, 90,5, 90,5, 90,5,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['y'] = [-10,5, 90,6, 90,6, 90,6, 20,10,   
-1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 14,20, 8,14, 0,4]
data_range['z'] = [50,10, 7,7, 90,5, 90,5, 90,5,  
-1,1, -1,1, -1,1, -30,-5, 100,120, -5,5, 14,20, 8,14, 0,4] #moving




def generate_flex_data(mean, var):  
    return float(rng.normal(loc=mean, scale=var, size=1))

def generate_imu_data(dir_min, dir_max): 
    dist = dir_max - dir_min
    norm = max(0,dir_min)
    return np.random.rand()*dist - norm 

def generate_one_vector(label, outlier_rate):
    data = data_range[label]
    vector = list()
    for i in range(0,len(data), 2):
        #0.3% chance of outliers
        if(np.random.rand() < outlier_rate):
            vector.append(np.random.rand()*100)
            continue
        #fingers
        if(i<10):
            vector.append(generate_flex_data(data[i],data[i+1]))
        #imus
        else:
            vector.append(generate_imu_data(data[i],data[i+1]))
  
    return vector



def generate_data(filename, sample_count = 100, outlier_rate=0.003):
    
    f = open(filename,"w")
    cr = csv.writer(f, delimiter=" ",lineterminator="\n")
    cr.writerow(['label', "finger1","finger2","finger3","finger4","finger5",
    "acc1","acc2","acc3","gyro1","gyro2","gyro3","mag1","mag2","mag3"])#headre

    
    for i in range(1, 1+len(labels)): #TODO, change
        label = chr(i+96)
        for j in range(0,sample_count):
            final_vector=[i]
            final_vector+=(generate_one_vector(label,outlier_rate)) 
            cr.writerow(final_vector)
    f.close()




if __name__ == "__main__":
    generate_data("simulated_train_data.csv")
    generate_data("simulated_test_data.csv",50,0.05)