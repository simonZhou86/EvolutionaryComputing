import torch
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
sys.path.append("/home/simonz/CISC455")

'''
This script contains some helper functions for proj.py

'''


def sort(file):
	'''
	this function sortes the data for a given directory
	'''
	inidi = int(file.split(".")[0])
	return inidi


def load_data(cur_dir, width, height, y_lab):
	'''
	this function returns the data and labels in torch Tensor format
	cur_dir: data directory
	width, height: image width, height
	y_lab: training labels
	'''
	# data tensor
	data = torch.empty(0, 3, width, height)
	# label tensor
	labels = torch.empty(0)
	sorted_path = sorted([file for file in os.listdir(cur_dir)], key=sort)
	#print(sorted_path)
	for ind, f in enumerate(sorted_path):
		# reshape the image to 256*256*3
		crop = Image.open(cur_dir+"/"+f).convert('RGB').resize((256, 256))
		crop = np.array(crop, dtype=np.float32)
		# normalize data to [0,1]
		crop = crop / 255.
		print("file: {}, shape: {}".format(f, crop.shape))
		print("file: {}, max: {}, min: {}".format(f, np.max(crop), np.min(crop)))
		crop = torch.from_numpy(crop)
		# reshape to channel first data
		crop = np.transpose(crop,(2,1,0))
		print("reshape: ", crop.shape)
		crop = crop.reshape(1, *crop.shape)
		label = y_lab[0][ind]
		label = torch.Tensor([int(label)])
		# store data and corresponding label
		data = torch.cat((data, crop), dim=0)
		labels = torch.cat((labels, label))
	return data, labels


def processing(file):
	'''
	this function computes the class weighs
	file: training label file in csv
	'''
	y_train = pd.read_csv(file)["infection"]
	negative, positive = np.bincount(y_train)
	# compute weights
	class_0_weight = (1 / negative)*(negative + positive)/2.0 # weight of class 1
	class_1_weight = (1 / positive)*(negative + positive)/2.0 # weight of class 2
	label = np.array(y_train).reshape(1,-1)
	return label, [class_0_weight, class_1_weight]

# if __name__ == "__main__":
# 	file = "/home/simonz/CISC455/y_train_covid.csv"
# 	y_lab, class_weights = processing(file)
# 	cur_dir = "/home/simonz/CISC455/train_dat"
# 	width = height = 256
# 	x_train, x_lab = load_data(cur_dir, width, height, y_lab)
# 	print(x_train.shape, x_lab.shape)
# 	torch.save(x_train, "/home/simonz/CISC455/train_dat455.pt")
# 	torch.save(x_lab, "/home/simonz/CISC455/train_lab455.pt")




