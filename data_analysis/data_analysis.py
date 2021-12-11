import numpy as np
import matplotlib.pyplot as plt

 
def first():
    filename_fake = "../fake_data/simulated_train_data.csv"
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
    plt.show()
 

    return new_data
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


def plot_point_clouds(letter1, letter2):
    fn1 = "../real_data_nov5_window3/" + letter1 + ".npy"
    fn2 = "../real_data_nov5_window3/" + letter2 + ".npy"
    data1 = np.load(fn1)[:, :14]
    data2 = np.load(fn2)[:, :14]

    fn1 = "../rachel_roommates/" + letter1 + ".npy"
    fn2 = "../rachel_roommates/" + letter2 + ".npy"
    data1 = np.vstack((data1, np.load(fn1)[:, :14]))
    data2 = np.vstack((data2, np.load(fn2)[:, :14]))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c="b", label=letter1)
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="r", label=letter2)
    ax.set_title(letter1.upper() + " vs. " + letter2.upper() + " (Thumb, Pointer, Middle)")
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("123_point_cloud.png")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(data1[:, 3], data1[:, 4], c="b", label=letter1)
    ax.scatter(data2[:, 3], data2[:, 4], c="r", label=letter2)
    ax.set_title(letter1.upper() + " vs. " + letter2.upper() + " (Ring, Pinky)")
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("45_point_cloud.png")

 
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data1[:, 5], data1[:, 6], data1[:, 7], c="b", label=letter1)
    ax.scatter(data2[:, 5], data2[:, 6], data2[:, 7], c="r", label=letter2)
    ax.set_title(letter1.upper() + " vs. " + letter2.upper() + " (Accelerometer X, Y, Z)")
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("acc_point_cloud.png")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data1[:, 8], data1[:, 9], data1[:, 10], c="b", label=letter1)
    ax.scatter(data2[:, 8], data2[:, 9], data2[:, 10], c="r", label=letter2)
    ax.set_title(letter1.upper() + " vs. " + letter2.upper() + " (Gyroscope X, Y, Z)")
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("gyro_point_cloud.png")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data1[:, 11], data1[:, 12], data1[:, 13], c="b", label=letter1)
    ax.scatter(data2[:, 11], data2[:, 12], data2[:, 13], c="r", label=letter2)
    ax.set_title(letter1.upper() + " vs. " + letter2.upper() + " (Magnetometer X, Y, Z)")
    plt.legend(loc='upper right')
    plt.savefig("mag_point_cloud.png")

def generate_sys_speed_vs_accuracy():
    required_consecutive = [11, 10, 9, 8, 7, 6]
    success_rate = [15/25, 14/23, 12/20, 14/20, 16/28, 9/28]
    plt.figure()
    plt.plot(required_consecutive, success_rate)
    plt.title("Required Consecutive Predictions vs. Accuracy")
    plt.xlabel("Required Consecutive Predictions")
    plt.ylabel("Accuracy")
    plt.savefig("sys_speed_vs_acc.png")

def generate_user_speed_vs_accuracy():
    signing_rate = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    accuracy = [17/49, 16/34, 14/20, 12/20, 12/17, 15/22, 2/20]
    plt.figure()
    plt.plot(signing_rate, accuracy)
    plt.title("User Signing Rate vs. Accuracy")
    plt.xlabel("User Signing Rate")
    plt.ylabel("Accuracy")
    plt.savefig("user_speed_vs_acc.png")


def window3_data():
    paths = ['real_data_nov5_window3/','rachel_roommates/','data_sz_friends/']
    # paths = ['rachel_roommates/']
    letters = ['r','u','v','w']
    data = list()
    for l in letters:
        temp = np.empty( (0,42 ))
        for p in paths:
            l_data = np.load("../"+p+"/"+l+".npy")

            l_data = remove_outliers(l_data)
            temp = np.concatenate( (temp, l_data), axis=0)
        # data[l] = temp 
        data.append(temp)
    print(temp.shape)
    fingers = ['thumb','index','middle','ring','pinky']
    for i in range(0,5):

        plt.hist( [ data[0][:,i], data[1][:,i],data[2][:,i],data[3][:,i]], 
            label=letters, bins=20)

        plt.legend(loc='upper right')
        plt.title(fingers[i])
        plt.xlabel("bend angle")
        plt.show() 
 

def see_data_amounts():
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    labels = letters + ['space','transition']
    total = {}
    for path in ["../dec7_rachel/", "../dec7_sophia/", "../dec7_stephanie/"]:
        amounts = []
        for lab in labels:
            amt = len(np.load(path + lab + ".npy"))
            amounts.append(amt)
            if (lab in total):
                total[lab] += amt
            else:
                total[lab] = amt
        plt.bar(labels, amounts)
        plt.show()
    amounts = []
    for lab in labels:
        amounts.append(total[lab])
    plt.bar(labels, amounts)
    plt.show()

see_data_amounts()