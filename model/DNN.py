import backbones
import heads

import tensorflow as tf
import numpy as np
import sys

def create_DNN(input_shape, backbone, head, num_key_points):
	inputs=tf.keras.layers.Input(shape=input_shape)	
	model_name = ''
	if backbone not in list(backbones.possible_backbones.keys()):
		print('backbone is %s and should be in :'%(backbone))
		print(list(backbones.possible_backbones.keys()))
		sys.exit()
	if head not in list(heads.possible_heads.keys()):
		print('head is %s and should be in :'%(head))
		print(list(heads.possible_heads.keys()))
		sys.exit()
	x = backbones.possible_backbones[backbone](x=inputs)
	x = heads.possible_heads[head](x, num_key_points, input_shape[0])
	DNN = tf.keras.models.Model(inputs, x, name=model_name)
	return DNN

if __name__ == '__main__':
	for head in list(heads.possible_heads.keys()):
		for backbone in list(backbones.possible_backbones.keys()):
			DNN = create_DNN(input_shape=(224,224,3), backbone=backbone, head = head, num_key_points = 16)
			num_params = np.sum([np.prod(v.shape) for v in DNN.trainable_variables]) +\
				np.sum([np.prod(v.shape) for v in DNN.non_trainable_variables])
			print('Model [%s + %s] : %s parameters'%(backbone, head, '{:.1E}'.format(num_params)))