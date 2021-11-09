import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from train.transfo import transform_labels_heatmaps
from blocks import up_sample

def create_scalar_heads(x, num_key_points, scale):
	y = tf.keras.layers.Flatten()(x)
	outputs = scale * tf.keras.layers.Dense(
		2*num_key_points,
		activation='sigmoid',
		use_bias=False,
		kernel_initializer='he_normal')(y)
	return outputs

def create_heatmap_heads(x,num_key_points,scale):

	output = scale*Conv2D(filters=num_key_points,kernel_size=3,strides=(1,1),activation="sigmoid",padding="same")(x)
	output = tf.keras.layers.Reshape((num_key_points,output.shape[1],output.shape[1],1))(output)
	
	return output

def create_heatmap_heads1(x,num_key_points,scale):

	# x = Conv2D(filters=num_key_points*2,kernel_size=3,strides=(1,1),activation="sigmoid",padding="same")(x)
	output = Conv2D(filters=num_key_points,kernel_size=1,strides=(1,1),activation="sigmoid")(x)
	output = tf.keras.layers.Reshape((num_key_points,output.shape[1],output.shape[1],1))(output)
	
	return output

def create_identity_heads(x,num_key_points,scale):
	return x

possible_heads = {
	'scalar' : create_scalar_heads,
	'heatmap' : create_heatmap_heads1,
	#'heatmap1' : create_heatmap_heads1,
	#'identity' : create_identity_heads,
}