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
THUMBS_UP = 1
THUMBS_DOWN = 2
PEACE = 3
LOVE = 4 
STOP = 5
data_range = dict()
data_range[THUMBS_UP] = [91,6, 91,6, 91,6, 91,6, 5,5, -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, 16,35, 11,35, 0,4]
data_range[THUMBS_DOWN] = [91,6, 91,6, 91,6, 91,6, 5,5,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -0.3,11, 5.5,17, 4,8]
data_range[PEACE] = [50,10, 7,7, 7,7, 90,5, 90,5,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range[LOVE] = [5,5, 7.5,7.5, 90,6, 90,6, 20,10,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]
data_range[STOP] = [7.5,7.5, 7.5,7.5, 7.5,7.5, 7.5,7.5, 7.5,7.5,  -1,1, -1,1, -1,1, -5,5, -5,5, -5,5, -8,0, -6,0, 0,4]


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

 
    for i in range(1, 6):
        for j in range(0,sample_count):
            final_vector=[i]
            final_vector+=(generate_one_vector(i,outlier_rate)) 
            cr.writerow(final_vector)
    f.close()




if __name__ == "__main__":
    generate_data("simulated_train_data.csv")
    generate_data("simulated_test_data.csv",50,0.05)