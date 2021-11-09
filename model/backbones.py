from model.blocks import residual_block_1, residual_block_with_pooling
from blocks import residual_block, inverted_residual_block, conv_block, up_sample

import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D
import sys


def create_vgg(x):
	x = tf.keras.layers.Conv2D(16,
		kernel_size=7,
		strides=4,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=2,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	return x

def create_identity(x):
	return x


def create_resNet_hourglass(x):

	#x = conv_block(x,num_filters=16,kernel_size=7,strides=2,activation='relu')
	#x = residual_block_with_pooling(x,k=1,kernel_size=3,num_filters=16)

	# Encoder
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=32)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=64)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=128)
	x = MaxPool2D(pool_size=(2,2))(x)

	# Decoder
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=128)
	x = up_sample(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=64)
	x = up_sample(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=32)
	x = up_sample(x)

	return x


def create_resNet_hourglass1(x):


	# Encoder
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=32)
	x = tf.keras.layers.Activation('relu')(x)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)
	x = MaxPool2D(pool_size=(2,2))(x)

	# Decoder
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)
	x = up_sample(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = up_sample(x)
	x = residual_block_1(x,k=1,kernel_size=3,num_filters=32)
	x = tf.keras.layers.Activation('relu')(x)
	x = up_sample(x)

	return x

def create_resNet_hourglass2(x):


	# Encoder
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=32)
	x = tf.keras.layers.Activation('relu')(x)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = MaxPool2D(pool_size=(2,2))(x)
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)
	#x = MaxPool2D(pool_size=(2,2))(x)

	# Decoder
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)
	x = up_sample(x)
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = up_sample(x)
	x = residual_block_1(x,k=2,kernel_size=3,num_filters=32)
	x = tf.keras.layers.Activation('relu')(x)
	#x = up_sample(x)

	return x


def create_stacked_hourglass(x):

	return create_resNet_hourglass(create_resNet_hourglass(x))

def create_stacked_hourglass1(x):

	return create_resNet_hourglass1(create_resNet_hourglass1(x))

def create_stacked_hourglass2(x):

	return create_resNet_hourglass2(create_resNet_hourglass2(x))

def create_resNet(x):

	x = conv_block(x,num_filters=64,kernel_size=7,strides=2,activation='relu')
	x = residual_block_with_pooling(x,k=1,kernel_size=3,num_filters=16)
	x = residual_block(x,k=1,kernel_size=3,num_filters=32)
	x = residual_block(x,k=1,kernel_size=3,num_filters=64)
	x = residual_block(x,k=1,kernel_size=3,num_filters=128)

	x = up_sample(x)
	x = conv_block(x,num_filters=128,kernel_size=3,strides=1,activation="relu")
	x = up_sample(x)
	x = conv_block(x,num_filters=64,kernel_size=3,strides=1,activation="relu")
	x = up_sample(x)
	x = conv_block(x,num_filters=32,kernel_size=3,strides=1,activation="relu")
	x = up_sample(x)
	x = conv_block(x,num_filters=16,kernel_size=3,strides=1,activation="relu")
	x = up_sample(x)

	return x

def create_resNet1(x):

	x = conv_block(x,num_filters=16,kernel_size=7,strides=2,activation='relu')
	x = residual_block_with_pooling(x,k=1,kernel_size=3,num_filters=16)

	x1 = residual_block(x,k=1,kernel_size=3,num_filters=32)
	x2 = residual_block(x1,k=1,kernel_size=3,num_filters=32)
	x3 = residual_block(x2,k=1,kernel_size=3,num_filters=32)


	x2_ = up_sample(x2)
	x3_ = up_sample(up_sample(x3))
	

	concat = tf.keras.layers.Concatenate()([x1,x2_,x3_])

	output = up_sample(up_sample(concat))

	return output

def create_resNet2(x):

	x = conv_block(x,num_filters=16,kernel_size=7,strides=2,activation='relu')
	x = residual_block_with_pooling(x,k=2,kernel_size=3,num_filters=16)

	x1 = residual_block(x,k=2,kernel_size=3,num_filters=32)
	x2 = residual_block(x1,k=2,kernel_size=3,num_filters=32)
	x3 = residual_block(x2,k=2,kernel_size=3,num_filters=32)


	x2_ = up_sample(x2)
	x3_ = up_sample(up_sample(x3))
	
	concat = tf.keras.layers.Concatenate()([x1,x2_,x3_])

	output = up_sample(up_sample(concat))

	return output

possible_backbones = {
	'VGG':create_vgg,
	'Identity':create_identity,
	'ResNet':create_resNet,
	'ResNet1':create_resNet1,
	'ResNet2':create_resNet2,
	'ResNetHG':create_resNet_hourglass,
	'ResNetHG1':create_resNet_hourglass1,
	'StackedHG':create_stacked_hourglass,
	'StackedHG1':create_stacked_hourglass1,
	'StackedHG2':create_stacked_hourglass2,
}
