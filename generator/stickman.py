from keypoints import key_points_generator, draw_a_stickman
from background import background_generator
import colors
import numpy as np
import cv2
import sys
import time
import tensorflow as tf

class stick_man_generator(tf.keras.utils.Sequence):
	def __init__(self, batch_size = 2, set_of_data = 'train', p_circles=0, p_squares=0, p_real=0, input_shape = (224,224,3)):
		self.set = set_of_data
		self.possible_colors = colors.possible_colors
		self.input_shape = input_shape
		self.batch_size = batch_size
		self.key_points = key_points_generator(batch_size=batch_size, set_of_data=set_of_data)
		self.background = background_generator(
			batch_size=batch_size, 
			p_circles = p_circles, 
			p_squares = p_squares, 
			p_real = p_real,
			length=self.__len__(),
			input_shape=self.input_shape)
		
	def on_epoch_end(self):
		self.key_points.on_epoch_end()

	def data_generation(self, index):
		KP = self.key_points.__getitem__(index=index)
		stickman_colors = np.random.choice(len(self.possible_colors), len(KP), replace=True)
		self.background.stickman_colors = stickman_colors
		images = self.background.__getitem__(index=index)
		KP_x = np.copy(KP[:,::2])
		KP_y = np.copy(KP[:,1::2])
		KP_x = (KP_x - np.min(KP_x)) / (np.max(KP_x) - np.min(KP_x))
		KP_y = (KP_y - np.min(KP_y)) / (np.max(KP_y) - np.min(KP_y))
		KP_x = KP_x * self.input_shape[0] / np.max(KP_x)
		KP_y = KP_y * self.input_shape[1] / np.max(KP_y)
		for b in range(self.batch_size):
			images[b,...] = draw_a_stickman(images[b,...], X=KP_x[b,:], Y=KP_y[b,:], color = self.possible_colors[stickman_colors[b]], input_shape = self.input_shape[0])
		KP[:,::2] = KP_x
		KP[:,1::2] = KP_y
		images = images/255.
		return images, KP

	def __len__(self):
		return self.key_points.__len__()

	def __getitem__(self, index):
		data = self.data_generation(index)
		return data

def run_visualisation():
	print('test data generator')
	test = stick_man_generator(batch_size = 2, set_of_data = 'train', p_circles=0.5, p_squares=0.5, p_real=0.5, input_shape = (224,224,3))
	duration = 0
	to_stop = False
	for i in range(test.__len__()):
		print('\rvisualisation : batch (2x224x224x3) %i/%i         '%(i+1, test.__len__()), end='')
		tmp = time.time()
		imgs=test.__getitem__(i)[0]
		duration += time.time() - tmp
		for b in range(len(imgs)):
			cv2.imshow('',np.uint8(imgs[b]))
			k = cv2.waitKey(33)
			if k==27:	# Esc key to stop
				to_stop = True
			elif k==-1:  # normally -1 returned,so don't print it
				continue
			elif k==32: # Esc space to pause
				print('\rvisualisation : batch (2x224x224x3) %i/%i (paused)'%(i+1, test.__len__()), end='') 
				k = cv2.waitKey(33)
				while(k != 32):
					k = cv2.waitKey(33)
					continue
		if to_stop:
			break
	print()
	cv2.destroyWindow('')
	print('average time for batch creation : ', duration/test.__len__())
	test = stick_man_generator(batch_size = 128, set_of_data = 'train', p_circles=0.5, p_squares=0.5, p_real=0.5, input_shape = (1920,1080,3))
	duration = 0
	for i in range(test.__len__()):
		print('\rvisualisation : batch (128x1920x1080x3) %i/%i'%(i+1, test.__len__()), end='')
		tmp = time.time()
		imgs=test.__getitem__(i)[0]
		duration += time.time() - tmp
	print()
	print('average time for batch creation : ', duration/test.__len__())

if __name__ == '__main__':
	run_visualisation()
