import numpy as np

num_captures = 1
num_data = 500

filename = "generated_data.npy"

'''
data ranges

According to a paper, the resting position is actually 
about 30 degrees, so anything that is supposed to be 
"straight" I allow for up to 30 degrees

Also, giving larger ranges of angles to account for different
sized hands as well as capture how loosely one might hold 
each gesture

'''
variances = [5, 5, 5, 5, 5]
ranges = [[[0, 30], [80,100], [80,100], [80,100], [80,100]], # thumbs up
           [[0, 30], [80,100], [80,100], [80,100], [80,100]], # thumbs down
           [[50, 70], [0, 30], [0, 30], [60, 90], [60, 90]], # peace sign
           [[0, 30], [0, 30], [60, 90], [60, 90], [0, 30]], # love sign
           [[0, 30], [0, 30], [0, 30], [0, 30], [0, 30]], # stop sign
           [[[]]]
           ]

'''
generate data!

1. grab random float within the range for each component
2. add on noise from a normal distribution centered at 0
    - variance 5 degrees for flex sensors
'''
data = []
labels = []
for i in range(500):
    gesture = np.random.randint(0, len(variances))
    labels.append(gesture)
    data_range = ranges[gesture]
    point = []
    for j in range(len(variances)):
        point.append(np.random.random()*(data_range[j][1] - data_range[j][0])
                     + data_range[j][0] 
                     + np.random.normal(0, variances[j]))   
    data.append(point) 

data = np.array(data)
labels = np.array(labels)

np.save(filename, {"data":data, "labels":labels})