import os
import cv2
import numpy as np
from tensorflow.python.keras.backend import dtype
from train.metrics import softmax2D_coords, max_coords
import matplotlib.pyplot as plt
import time


def evaluate_on_a_set(DNN, image_shape, eval_set):
	preds = []
	errors = []
	print("EVALUATE ON A SET")
	print(os.listdir('..'))
	if 'Dataset' in os.listdir('..'):
		if 'Test' + str(eval_set) in os.listdir('../Dataset'):
			is_labelled = False
			if 'test' + str(eval_set) + '_labels.npy' in os.listdir('../Dataset'):
				labels = np.load('../Dataset/' + 'test' + str(eval_set) + '_labels.npy')
				is_labelled = True
			images = ['../Dataset/Test' + str(eval_set) + '/' + e for e in os.listdir('../Dataset/Test' + str(eval_set)) if '.png' in e]
			images.sort()
			for cpt, img in enumerate(images):
				print('\rtesting on set %i/4 : %i/%i'%(eval_set, cpt+1, len(images)), end='')
				img_ = cv2.imread(img, cv2.IMREAD_UNCHANGED)
				img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), (image_shape, image_shape)) / 255
				pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape
				
				#print(len(pred.shape))

				# Heatmap prediction
				if len(pred.shape)==5:
					#print("IN")
					pred = softmax2D_coords(pred)


				if is_labelled:
					KP=labels[cpt]
					KP_x = np.copy(KP[::2]) / img_.shape[0]
					KP_y = np.copy(KP[1::2]) / img_.shape[1]
					KP[::2] = KP_x
					KP[1::2] = KP_y
					errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
				else:
					#print("OK CA FONCTIONNE",preds)
					preds.append(pred)
			print()
		#print("TEST")
		np.save("PredsTEST"+str(eval_set),preds)
		
		if is_labelled:
			return np.mean(errors)
		return None


def eval_DNN(DNN, image_shape):
	return (evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=1),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=2),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=3),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=4))


	
def evaluate_on_a_val_set(DNN, image_shape, eval_set):
	preds = []
	errors = []

	images = eval_set[0]
	labels = eval_set[1]

	

	for cpt, img in enumerate(images):
		
		img_init = img
		img = cv2.resize(img, (image_shape, image_shape)) / 255
		pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape

		"""
		i_ = pred[0,0,:,:,:]/np.max(pred[0,0,:,:,:])
		print(i_.shape)
		cv2.imshow("heatmap",cv2.resize(i_,(512,512)))
		cv2.waitKey(0)
		"""


		# Heatmap prediction
		if len(pred.shape)==5:
			temp = []
			for n_heatmap in range(pred.shape[1]):
				coord = max_coords(pred[0,n_heatmap,:,:,:])
				temp.append(coord[0] / img_init.shape[0])
				temp.append(coord[1] / img_init.shape[1])

			pred = np.array([temp],dtype=float) 

		KP=labels[cpt].astype(float)
		KP_x = np.copy(KP[::2]) / img_init.shape[0]
		KP_y = np.copy(KP[1::2]) / img_init.shape[1]
		KP[::2] = KP_x
		KP[1::2] = KP_y

		# print(KP)
		# print(pred)
		# print(np.abs(KP - pred))
		# print(np.abs(KP - pred)[0])
		# print(np.abs(KP - pred)[0][KP > 0])

		errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
		preds.append(pred)
		#time.sleep(3)
		
	return np.mean(errors)

