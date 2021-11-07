import numpy as np
import matplotlib.pyplot as plt


'''filename_fake = "../fake_data/simulated_train_data.csv"
filename_real = "../real_data/l.npy"

fake_data_full = np.genfromtxt(filename_fake, delimiter=" ", skip_header=1, dtype=float)
fake_data = fake_data_full[1100:1200, :]
real_data = np.load(filename_real, allow_pickle=True)

print("fake data shape:", fake_data.shape)
print("real data shape:", real_data.shape)

# 0, 1, 4
plt.xlabel("Thumb")
plt.ylabel("Pointer")
plt.title("Real Data: Thumb vs. Pointer vs. Pinky")
plt.scatter(real_data[:, 0], real_data[:, 1], s=real_data[:, 4]*5)
plt.savefig("real_014.png")
plt.show()

plt.xlabel("Thumb")
plt.ylabel("Pointer")
plt.title("Fake Data: Thumb vs. Pointer vs. Pinky")
plt.scatter(fake_data[:, 0], fake_data[:, 1], s=fake_data[:, 4]*5)
plt.savefig("fake_014.png")
plt.show()

#2, 8, 10
plt.xlabel("Middle")
plt.ylabel("AccelerometerZ")
plt.title("Real Data: Middle vs. AccelerometerZ vs. GyroscopeY")
plt.scatter(real_data[:, 2], real_data[:, 8], s=real_data[:, 10]*5)
plt.savefig("real_2810.png")
plt.show()

plt.xlabel("Middle")
plt.ylabel("AccelerometerZ")
plt.title("Fake Data: Middle vs. AccelerometerZ vs. GyroscopeY")
plt.scatter(fake_data[:, 2], fake_data[:, 8], s=fake_data[:, 10]*5)
plt.savefig("fake_2810.png")
plt.show()

# 11, 12, 13
plt.xlabel("GyroscopeX")
plt.ylabel("GyroscopeY")
plt.title("Real Data: Magnetometer")
plt.scatter(real_data[:, 11], real_data[:, 12], s=real_data[:, 13]*10)
plt.savefig("real_111213.png")
plt.show()

plt.xlabel("GyroscopeX")
plt.ylabel("GyroscopeY")
plt.title("Fake Data: Magnetometer")
plt.scatter(fake_data[:, 11], fake_data[:, 12], s=fake_data[:, 13]*10)
plt.savefig("fake_111213.png")
plt.show()'''

def oct26_pj():
    p = np.load("../real_data_oct26/p.npy")
    j = np.load("../real_data_oct26/j.npy")

    plt.figure()
    plt.hist(p[:, 5])
    plt.savefig("oct26_p_accx.png")

    plt.figure()
    plt.hist(p[:, 6])
    plt.savefig("oct26_p_accy.png")

    plt.figure()
    plt.hist(p[:, 7])
    plt.savefig("oct26_p_accz.png")

    plt.figure()
    plt.hist(j[:, 5])
    plt.savefig("oct26_j_accx.png")

    plt.figure()
    plt.hist(j[:, 6])
    plt.savefig("oct26_j_accy.png")

    plt.figure()
    plt.hist(j[:, 7])
    plt.savefig("oct26_j_accz.png")

    plt.figure()
    plt.hist(p[:, 8])
    plt.savefig("oct26_p_gyrx.png")

    plt.figure()
    plt.hist(p[:, 9])
    plt.savefig("oct26_p_gyry.png")

    plt.figure()
    plt.hist(p[:, 10])
    plt.savefig("oct26_p_gyrz.png")

    plt.figure()
    plt.hist(j[:, 8])
    plt.savefig("oct26_j_gyrx.png")

    plt.figure()
    plt.hist(j[:, 9])
    plt.savefig("oct26_j_gyry.png")

    plt.figure()
    plt.hist(j[:, 10])
    plt.savefig("oct26_j_gyrz.png")


def oct26_uw():

    u = np.load("../real_data_oct26/u.npy")
    w = np.load("../real_data_oct26/w.npy")
    dataset = 'oct26_uw'
    
    #fingers 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # plt.hist( [w[:, 2],u[:, 2]], label=['u','w'])
    # # plt.legend(loc='upper right') 
     
    # plt.scatter( u[:,0], u[:,1], u[:,2], c='b',marker='o', label='u')
    # plt.scatter( w[:,0], w[:,1], w[:,2], c='r', marker='x', label='w')
    # plt.xlim(100,300)
    # plt.ylim(100,300)  
    # plt.savefig("oct26_uw_fing1-3.png")
 

    # plt.figure()  
    # plt.hist([u[:,0],w[:,0]]) 
    # plt.figure()  
    # plt.hist([u[:,1],w[:,1]]) 
    # plt.figure()  
    # plt.hist([u[:,2],w[:,2]]) 
    plt.figure()  
    plt.hist([u[:,2],w[:,2]]) 

    plt.savefig("oct26_uw_fing3_hist.png")
    plt.figure()
    plt.scatter( u[:,2], [0]*len(u[:,2]),   c='b',marker='o', label='u')
    plt.scatter( w[:,2], [1]*len(w[:,2]),   c='r', marker='x', label='w')
    # plt.xlim(0,300)
    # plt.ylim(0,300)   
    plt.savefig("oct26_uw_fing3_scatter.png")

    #acc

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(u[:, 5], u[:, 6], u[:, 7], c = 'b', marker='o',label='u')
    ax.scatter(w[:, 5], w[:, 6], w[:, 7], c = 'r', marker='x', label='w')
    ax.set_title("u vs. w accelerometer")
    plt.legend(loc='upper right') 
    plt.savefig(dataset + "_acc.png")

    #mag 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(u[:, 8], u[:, 9], u[:, 10], c = 'b', marker='o',label='u')
    ax.scatter(w[:, 8], w[:, 9], w[:, 10], c = 'r', marker='x', label='w')
    ax.set_title("u vs. w magnetometer")
    plt.legend(loc='upper right') 
    plt.savefig(dataset + "_mag.png")
  
    #gyro
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(u[:, 11], u[:, 12], u[:, 13], c = 'b', marker='o',label='u')
    ax.scatter(w[:, 11], w[:, 12], w[:, 13], c = 'r', marker='x', label='w')
    ax.set_title("u vs. w gyroscope")
    plt.legend(loc='upper right') 
    plt.savefig(dataset + "_gyro.png")

     

   
 


oct26_uw()