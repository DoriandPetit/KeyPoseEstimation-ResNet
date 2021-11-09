import numpy as np
import tensorflow as tf
import csv
import colors
import numpy as np
import cv2
import os
import sys

def clear_labels(labels):
	print('found %i labels' %(len(labels)))
	output = []
	for l in labels:
		if -1 not in l:
			output.append(l)
	print('kept %i labels' %(len(output)))
	output = np.array(output)
	return output
def get_annotations(set_of_data):
	print('\rgathering annotations : ...', end='')
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	if set_of_data == 'train':
		file = '/train_split.npy'
	elif set_of_data == 'val':
		file = '/train_split.npy'
	elif set_of_data == 'test':
		file = '/test_split.npy'
	labels = np.load(dname + file)
	labels=clear_labels(labels)
	# if set_of_data == 'train':
	# 	labels = labels[:7000]
	# elif set_of_data == 'val':
	# 	labels = labels[7000:]
	print('\rgathering annotations : done', end='')
	print()
	return labels

class key_points_generator(tf.keras.utils.Sequence):
	def __init__(self, batch_size = 2, set_of_data = 'train'):
		self.set = set_of_data
		self.batch_size = batch_size
		self.labels = get_annotations(set_of_data)
		self.indexes = np.arange(len(self.labels))
		self.on_epoch_end()

	def on_epoch_end(self):
		if self.set == 'train':
			np.random.shuffle(self.indexes)

	def data_generation(self, batch):
		return self.labels[batch]

	def __len__(self):
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		data = self.data_generation(batch)
		return data

def draw_a_stickman(img, X, Y, color, input_shape):
	thickness = max(int(np.round(4*input_shape/224)), 1)
	# right leg
	if X[0] > 0 and Y[0] > 0 and X[1] > 0 and Y[1] > 0:
		cv2.line(img, (int(np.round(X[0])), int(np.round(Y[0]))), (int(np.round(X[1])), int(np.round(Y[1]))), color, thickness=thickness)
	if X[1] > 0 and Y[1] > 0 and X[2] > 0 and Y[2] > 0:
		cv2.line(img, (int(np.round(X[1])), int(np.round(Y[1]))), (int(np.round(X[2])), int(np.round(Y[2]))), color, thickness=thickness)
	# left leg
	if X[3] > 0 and Y[3] > 0 and X[4] > 0 and Y[4] > 0:
		cv2.line(img, (int(np.round(X[3])), int(np.round(Y[3]))), (int(np.round(X[4])), int(np.round(Y[4]))), color, thickness=thickness)
	if X[4] > 0 and Y[4] > 0 and X[5] > 0 and Y[5] > 0:
		cv2.line(img, (int(np.round(X[4])), int(np.round(Y[4]))), (int(np.round(X[5])), int(np.round(Y[5]))), color, thickness=thickness)
	# hips
	if X[3] > 0 and Y[3] > 0 and X[6] > 0 and Y[6] > 0:
		cv2.line(img, (int(np.round(X[3])), int(np.round(Y[3]))), (int(np.round(X[6])), int(np.round(Y[6]))), color, thickness=thickness)
	if X[2] > 0 and Y[2] > 0 and X[6] > 0 and Y[6] > 0:
		cv2.line(img, (int(np.round(X[2])), int(np.round(Y[2]))), (int(np.round(X[6])), int(np.round(Y[6]))), color, thickness=thickness)
	# torso
	if X[6] > 0 and Y[6] > 0 and X[7] > 0 and Y[7] > 0:
		cv2.line(img, (int(np.round(X[6])), int(np.round(Y[6]))), (int(np.round(X[7])), int(np.round(Y[7]))), color, thickness=thickness)
	if X[7] > 0 and Y[7] > 0 and X[8] > 0 and Y[8] > 0:
		cv2.line(img, (int(np.round(X[7])), int(np.round(Y[7]))), (int(np.round(X[8])), int(np.round(Y[8]))), color, thickness=thickness)
	if X[8] > 0 and Y[8] > 0 and X[9] > 0 and Y[9] > 0:
		cv2.line(img, (int(np.round(X[8])), int(np.round(Y[8]))), (int(np.round(X[9])), int(np.round(Y[9]))), color, thickness=thickness)
	# arms
	if X[10] > 0 and Y[10] > 0 and X[11] > 0 and Y[11] > 0:
		cv2.line(img, (int(np.round(X[10])), int(np.round(Y[10]))), (int(np.round(X[11])), int(np.round(Y[11]))), color, thickness=thickness)
	if X[11] > 0 and Y[11] > 0 and X[12] > 0 and Y[12] > 0:
		cv2.line(img, (int(np.round(X[11])), int(np.round(Y[11]))), (int(np.round(X[12])), int(np.round(Y[12]))), color, thickness=thickness)
	if X[12] > 0 and Y[12] > 0 and X[13] > 0 and Y[13] > 0:
		cv2.line(img, (int(np.round(X[12])), int(np.round(Y[12]))), (int(np.round(X[13])), int(np.round(Y[13]))), color, thickness=thickness)
	if X[13] > 0 and Y[13] > 0 and X[14] > 0 and Y[14] > 0:
		cv2.line(img, (int(np.round(X[13])), int(np.round(Y[13]))), (int(np.round(X[14])), int(np.round(Y[14]))), color, thickness=thickness)
	if X[14] > 0 and Y[14] > 0 and X[15] > 0 and Y[15] > 0:
		cv2.line(img, (int(np.round(X[14])), int(np.round(Y[14]))), (int(np.round(X[15])), int(np.round(Y[15]))), color, thickness=thickness)
	r = int(np.sqrt((X[8] - X[9])**2 + (Y[8] - Y[9])**2)/2)
	cv2.circle(img,(int((X[8] + X[9])/2), int((Y[8] + Y[9])/2)), r, tuple(color), -1)
	return img