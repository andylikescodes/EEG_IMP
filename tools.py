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


# Tools

def load(filepath):
	"""
	A handy program to extract several files of data for use.
	"""
	cube = loadmat(filepath)
	data = cube['DATA_CUBE']
	btn = cube['btn']
	auto = cube['auto']
	tAx = cube['tAx']
	SR = cube['SR']
	return data, btn, auto, tAx, SR

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






























