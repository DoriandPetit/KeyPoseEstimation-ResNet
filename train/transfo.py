import numpy as np
import sys

def transform_labels_heatmaps(label, shape):
	heatmaps = []
	for b in range(len(label)):
		tmp = []
		for h in range(int(len(label[b])/2)):
			tmp.append(np.expand_dims(np.outer(
				np.exp(-(np.arange(shape) - label[b,::2][h])**2), 
				np.exp(-(np.arange(shape) - label[b,1::2][h])**2)), axis = -1))
		heatmaps.append(tmp)
	return np.array(heatmaps)

def transform_labels_scalars(label, shape):
	return label