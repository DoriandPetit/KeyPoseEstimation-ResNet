from generator.stickman import stick_man_generator
import os
import cv2
import sys
import numpy as np

def generate_test_dataset():
	dataset = stick_man_generator(batch_size = 1, set_of_data = 'test', p_circles=0, p_squares=0, p_real=0, input_shape = (224,224,3))
	num_images = dataset.__len__()
	if 'Dataset' not in os.listdir('..'):
		os.mkdir('../Dataset')
		os.mkdir('../Dataset/Test1')
		os.mkdir('../Dataset/Test2')
		os.mkdir('../Dataset/Test3')
		os.mkdir('../Dataset/Test4')

	KPs = []
	for i in range(int(num_images/4)):
		print('\rcreating test set 1 : %i/%i'%(i+1, int(num_images/4)),end='')
		images,KP=dataset.__getitem__(index=i)
		KPs.append(KP[0])
		cv2.imwrite('../Dataset/Test1/' + str(i) + '.png',np.uint8(255*images[0]))
	np.save('../Dataset/test1_labels.npy', KPs)
	print('\rcreating test set 1 : --- done ---\n')
	dataset = stick_man_generator(batch_size = 1, set_of_data = 'test', p_circles=0.5, p_squares=0, p_real=0, input_shape = (224,224,3))
	KPs = []
	for i in range(int(num_images/4)):
		print('\rcreating test set 2 : %i/%i'%(i+1, int(num_images/4)),end='')
		images,KP=dataset.__getitem__(index=i+int(num_images/4))
		KPs.append(KP[0])
		cv2.imwrite('../Dataset/Test2/' + str(i) + '.png',np.uint8(255*images[0]))
	np.save('../Dataset/test2_labels.npy', KPs)
	print('\rcreating test set 2 : --- done ---\n')
	dataset = stick_man_generator(batch_size = 1, set_of_data = 'test', p_circles=0.75, p_squares=0.5, p_real=0, input_shape = (224,224,3))
	KPs = []
	for i in range(int(num_images/4)):
		print('\rcreating test set 3 : %i/%i'%(i+1, int(num_images/4)),end='')
		images,KP=dataset.__getitem__(index=i+2*int(num_images/4))
		KPs.append(KP[0])
		cv2.imwrite('../Dataset/Test3/' + str(i) + '.png',np.uint8(255*images[0]))
	np.save('../Dataset/test3_labels.npy', KPs)
	print('\rcreating test set 3 : --- done ---\n')
	dataset = stick_man_generator(batch_size = 1, set_of_data = 'test', p_circles=0.25, p_squares=0.25, p_real=1, input_shape = (224,224,3))
	KPs = []
	for i in range(int(num_images/4)):
		print('\rcreating test set 4 : %i/%i'%(i+1, int(num_images/4)),end='')
		images,KP=dataset.__getitem__(index=i+3*int(num_images/4))
		KPs.append(KP[0])
		cv2.imwrite('../Dataset/Test4/' + str(i) + '.png',np.uint8(255*images[0]))
	np.save('../Dataset/test4_labels.npy', KPs)
	print('\rcreating test set 4 : --- done ---\n')