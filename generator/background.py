from colors import possible_colors

import numpy as np
import cv2
import time
import tensorflow as tf
import os
import sys

def convert_one_image(image):
	new_image = np.copy(image[:,:,:3])
	im = image[:,:,-1]
	idx = np.argwhere(image[:,:,-1]<2)
	new_image[idx[:,0],idx[:,1],:] = 0
	return new_image

def convert_png_to_BGR(images):
	new_images = []
	for b in range(len(images)):
		new_images.append(convert_one_image(images[b]))
	return new_images

def get_possible_backgrounds():
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	#dname = "./generator"
	images_ad = [dname + '\\backgrounds\\' + elem for elem in os.listdir(dname + '\\backgrounds') if '.png' in elem]
	images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in images_ad]
	print(images_ad)
	images = convert_png_to_BGR(images)
	return images

def add_a_square(img,color):
	X=np.random.randint(low=0, high=img.shape[0]+1, size=(2))
	X.sort()
	x1, x2=X
	Y=np.random.randint(low=0, high=img.shape[1]+1, size=(2))
	Y.sort()
	y1, y2=Y
	cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=tuple(color), thickness=-1)
	return img

def add_random_squares(images, p, stickman_colors):
	global possible_colors
	for b in range(len(images)):
		colors = np.delete(np.copy(possible_colors), stickman_colors[b], axis = 0)
		colors=colors.tolist()
		while p > np.random.uniform():
			color = colors[np.random.choice(len(colors), 1, replace=False)[0]]
			images[b,...]=add_a_square(img=images[b,...], color=color)
	return images

def add_a_circle(img,color):
	x=np.random.randint(low=0, high=img.shape[0]+1)
	y=np.random.randint(low=0, high=img.shape[1]+1)
	cv2.circle(img,(x, y), 10, tuple(color), -1)
	return img

def add_random_circles(images, p, stickman_colors):
	global possible_colors
	for b in range(len(images)):
		colors = np.delete(np.copy(possible_colors), stickman_colors[b], axis = 0)
		colors=colors.tolist()
		while p > np.random.uniform():
			color = colors[np.random.choice(len(colors), 1, replace=False)[0]]
			images[b,...]=add_a_circle(img=images[b,...], color=color)
	return images

def extract_in_image(img, shape):
	W,H,_ = img.shape
	w,h,_ = shape
	if W-w > 0 and H-h > 0:
		x=np.random.randint(low=0, high=W-w)
		y=np.random.randint(low=0, high=H-h)
		return img[x:x+w,y:y+h,:]
	return img

def load_image(images, p, back):
	for b in range(len(images)):
		if p > np.random.uniform():
			background = back[int(np.random.randint(
						low=0, 
						high=len(back)))]
			if images[b,:,:,:].shape[0] < background.shape[0] and images[b,:,:,:].shape[1] < background.shape[1]:
				images[b,...]=extract_in_image(
					img=background, 
					shape=images[b].shape)
	return images

class background_generator(tf.keras.utils.Sequence):
	def __init__(self, batch_size=2, length=10000, p_circles=0, p_squares=0, p_real=0, input_shape=(224,224,3)):
		self.batch_size=batch_size
		self.p_circles=p_circles
		self.p_squares=p_squares
		self.len=length
		self.input_shape=input_shape
		self.stickman_colors = np.zeros(batch_size)
		self.on_epoch_end()
		self.p_real=p_real
		if self.p_real > 0:
			self.real_images=get_possible_backgrounds()

	def on_epoch_end(self):
		pass

	def data_generation(self):
		backgrounds=np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
		if self.p_real > 0:
			backgrounds=load_image(images=backgrounds, p=self.p_real, back=self.real_images)
		if self.p_squares > 0:
			backgrounds=add_random_squares(images=backgrounds, p=self.p_squares, stickman_colors=self.stickman_colors)
		if self.p_circles > 0:
			backgrounds=add_random_circles(images=backgrounds, p=self.p_circles, stickman_colors=self.stickman_colors)
		return backgrounds

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		return self.data_generation()