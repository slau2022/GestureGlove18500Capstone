import numpy as np
import matplotlib.pyplot as plt


filename_fake = "fake_data/simulated_train_data.csv"
filename_real = "real_data/a.npy"

fake_data_full = np.loadtxt(open(filename_fake), delimiter=" ")
fake_data = []
for i in range(fake_data_full.shape[0]):
    if fake_data_full[i, 0] == 2:
        break
    fake_data.append(fake_data_full[i, :])
fake_data = np.array(fake_data)

real_data = np.load(filename_real, allow_pickle=True)

