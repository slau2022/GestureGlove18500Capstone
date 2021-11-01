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

p = np.load("real_data_oct26/p.npy")
j = np.load("real_data_oct26/j.npy")

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