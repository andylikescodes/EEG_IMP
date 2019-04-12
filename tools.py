import numpy as np
import pandas as pd
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt

# Constants

SELECTED_CHANNELS = [13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 28, 29, 30, 31, 32,
 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
 48, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
 78, 79, 80, 81, 82, 83, 84, 88, 89, 90, 97, 98, 
 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
 119, 120, 121, 122, 123, 124, 125, 126, 133, 134,
 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
 145, 146, 147, 148, 149, 150, 175, 176, 177, 178,
 179, 180, 181, 182, 183, 184, 185, 186, 199, 200,
 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
 211, 212, 213, 220, 221, 222, 223, 224, 225, 226,
 227, 228, 229, 230, 231, 232, 233, 234, 247, 248, 
 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
 259, 260, 261, 262, 263, 264, 271, 272, 273, 274,
 275, 276, 277, 278, 279, 280, 281, 282]

ELECTRODE_MAPPING_X = [ 1,  2,  2,  1,  2,  3,  3,  4,  4,  3,  4,  3,  4,  5,  5,  4,  5,
        6,  6,  5,  5,  7,  6,  5,  6,  6,  6,  7,  8,  8,  7,  8,  7,  9,
       10, 10,  9,  1,  2,  3,  2,  3,  4,  5,  4,  4,  3,  5,  6,  6,  5,
        7,  8,  7,  6,  7,  8,  9, 10,  9,  8]

ELECTRODE_MAPPING_Y = [4, 4, 5, 5, 3, 3, 4, 4, 5, 5, 1, 2, 2, 3, 4, 3, 5, 5, 6, 6, 1, 2,
       1, 2, 2, 3, 4, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7,
       7, 8, 8, 8, 8, 9, 9, 6, 6, 7, 7, 8, 8, 6, 6, 7, 7]

MAG_CHANNELS = [ 14,  17,  20,  23,  29,  32,  38,  41,  44,  47,  56,  59,  62,
        65,  68,  71,  74,  77,  80,  83,  89,  98, 101, 104, 107, 110,
       113, 116, 119, 122, 125, 134, 137, 140, 143, 146, 149, 176, 179,
       182, 185, 200, 203, 206, 209, 212, 221, 224, 227, 230, 233, 248,
       251, 254, 257, 260, 263, 272, 275, 278, 281]

GRA1_CHANNELS = [ 12,  15,  18,  21,  27,  30,  36,  39,  42,  45,  54,  57,  60,
        63,  66,  69,  72,  75,  78,  81,  87,  96,  99, 102, 105, 108,
       111, 114, 117, 120, 123, 132, 135, 138, 141, 144, 147, 174, 177,
       180, 183, 198, 201, 204, 207, 210, 219, 222, 225, 228, 231, 246,
       249, 252, 255, 258, 261, 270, 273, 276, 279]

GRAD2_CHANNELS = [ 13,  16,  19,  22,  28,  31,  37,  40,  43,  46,  55,  58,  61,
        64,  67,  70,  73,  76,  79,  82,  88,  97, 100, 103, 106, 109,
       112, 115, 118, 121, 124, 133, 136, 139, 142, 145, 148, 175, 178,
       181, 184, 199, 202, 205, 208, 211, 220, 223, 226, 229, 232, 247,
       250, 253, 256, 259, 262, 271, 274, 277, 280]

# Single trial draw the frequency

def freq_bands_power(signal):

	current_signal_fft = abs(np.fft.fft(signal, n=500))
	f = np.fft.fftfreq(500, d=1/500)

	delta = sum(current_signal_fft[0:4])
	theta = sum(current_signal_fft[4:8])
	alpha = sum(current_signal_fft[8:14])
	beta = sum(current_signal_fft[15:31])
	low_gamma = sum(current_signal_fft[30:61])
	high_gamma = sum(current_signal_fft[61:101])

	return delta, theta, alpha, beta, low_gamma, high_gamma

def freq_img(trial, window_mask, plot = True, figsize=(10, 14)):

	delta_all_chs = []
	theta_all_chs = []
	alpha_all_chs = []
	beta_all_chs = []
	low_gamma_all_chs = []
	high_gamma_all_chs = []

	for i in range(len(trial)):
		signal = trial[i, window_mask]
		delta, theta, alpha, beta, low_gamma, high_gamma = freq_bands_power(signal)
		delta_all_chs.append(delta)
		theta_all_chs.append(theta)
		alpha_all_chs.append(alpha)
		beta_all_chs.append(beta)
		low_gamma_all_chs.append(low_gamma)
		high_gamma_all_chs.append(high_gamma)

	all_bands = {'delta': delta_all_chs, 
				 'theta': theta_all_chs, 
				 'alpha': alpha_all_chs,
				 'beta': beta_all_chs, 
				 'low_gamma': low_gamma_all_chs, 
				 'high_gamma': high_gamma_all_chs }

	
	imgs = []

	for band in all_bands:
		img = np.zeros((9,10))
		for i in range(len(ELECTRODE_MAPPING_X)):
			img[ELECTRODE_MAPPING_Y[i]-1, ELECTRODE_MAPPING_X[i]-1] = all_bands[band][i]

		imgs.append(img)

	imgs = np.stack(imgs)

	if plot == True:
		fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
		row_ind = 0
		col_ind = 0

		chs_names = list(all_bands.keys())
		for i in range(len(imgs)):
			ax[row_ind, col_ind].imshow(imgs[i])
			ax[row_ind, col_ind].set_title(chs_names[i])
			col_ind = col_ind + 1
			if (col_ind == 2):
				col_ind = col_ind - 2
				row_ind = row_ind + 1

	return imgs

def plot_freq_bands(imgs, figsize=(10, 14)):
	chs_names = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
	fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
	row_ind = 0
	col_ind = 0

	for i in range(len(imgs)):
		ax[row_ind, col_ind].imshow(imgs[i])
		ax[row_ind, col_ind].set_title(chs_names[i])
		col_ind = col_ind + 1
		if (col_ind == 2):
			col_ind = col_ind - 2
			row_ind = row_ind + 1
# Some util

def load(filename):
	cube = loadmat(filename)
	data = cube['DATA_CUBE']
	btn = cube['btn']
	auto = cube['auto']
	tAx = cube['tAx']
	SR = cube['SR']
	return data, btn, auto, tAx, SR

# The electrode mapping tasks

def process_channels():
	filenames = glob.glob('./data/*.mat')
	e_pos_files = glob.glob("./electrode_pos/*.lay")
	planar_2 = pd.read_csv(e_pos_files[0], sep=' ', header=None)
	planar_1 = pd.read_csv(e_pos_files[1], sep=' ', header=None)
	mag = pd.read_csv(e_pos_files[2], sep=' ', header=None)
	planar_2.loc[:, 'type'] = 'gra2'
	planar_1.loc[:, 'type'] = 'gra1'
	mag.loc[:, 'type'] = 'mag'
	all_chs = pd.concat([planar_2, planar_1, mag], axis=0).set_index(0)
	all_chs_selected = all_chs.loc[SELECTED_CHANNELS, :].reset_index()

	gra2 = all_chs_selected.loc[all_chs_selected['type'] == 'gra2', :]
	gra1 = all_chs_selected.loc[all_chs_selected['type'] == 'gra1', :]
	mag = all_chs_selected.loc[all_chs_selected['type'] == 'mag', :]
	return gra2, gra1, mag

def channel_mapping(df):
	n = df.shape[0]
	x_ls = []
	y_ls = []
	for i in range(n):
		plt.scatter(df[1], df[2])
		xx = df.iloc[i, 1]
		yy = df.iloc[i, 2]
		plt.plot(xx, yy, 'r+')
		plt.show()
		x = input('What is the x?')
		y = input('What is the y?')
		x_ls.append(x)
		y_ls.append(y)
	df.loc[:, 'x'] = x_ls
	df.loc[:, 'y'] = y_ls

	fig, ax = plt.subplots()
	ax.scatter(df[1], df[2])

	txts = []
	for i in list(df.index):
		txts.append(str(df.loc[i, 'x']) + ' ,' + str(df.loc[i, 'y']))

	for i, txt in enumerate(txts):
		ax.annotate(txt, (df.iloc[i, 1], df.iloc[i, 2]))

	return x_ls, y_ls

def plot_channels(df):
	fig, ax = plt.subplots()
	ax.scatter(df[1], df[2])

	txts = []
	for i in list(df.index):
		txts.append(str(df.loc[i, 'x']) + ' ,' + str(df.loc[i, 'y']))

	for i, txt in enumerate(txts):
		ax.annotate(txt, (df.iloc[i, 1], df.iloc[i, 2]))

if __name__ == '__main__':
	gra2, gra1, mag = process_channels()
	x_ls, y_ls = channel_mapping(mag)






























