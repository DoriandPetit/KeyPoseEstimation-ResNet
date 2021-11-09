import tensorflow as tf

def conv_block(x, num_filters, kernel_size=3, strides=2, activation='relu'):
	x = tf.keras.layers.Conv2D(num_filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=False,
			kernel_initializer='he_normal',
			kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	if activation is not None:
		x = tf.keras.layers.Activation(activation)(x)
	return x

def up_sample(x):
	return tf.image.resize(
		images=x, 
		size=(int(2 * x.shape[1]),int(2 * x.shape[2])), 
		method=tf.image.ResizeMethod.BILINEAR)

def depthwise_separable_conv(x, num_filters, kernel_size=3, strides=2, activation='relu'):
	x = tf.keras.layers.DepthwiseConv2D(
		kernel_size,
		strides=strides,
		padding='same',
		use_bias=False,
		depthwise_initializer={
		'class_name': 'VarianceScaling',
			'config': {
			'scale': 2.0,
			'mode': 'fan_out',
			'distribution': 'truncated_normal'}})(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation(activation)(x)
	x = tf.keras.layers.Conv2D(num_filters,
		kernel_size=1,
		strides=1,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	if activation is not None:
		x = tf.keras.layers.Activation(activation)(x)
	return x

def residual_block(x, k, kernel_size, num_filters):
	y = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=kernel_size, 
		strides=2, 
		activation='relu')
	for i in range(k-1):
		activation = 'relu'
		if i == k-2:
			activation=None
		y = conv_block(
			x=y, 
			num_filters=num_filters, 
			kernel_size=1, 
			strides=1, 
			activation=activation)
	x = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=1, 
		strides=2, 
		activation='relu')
	x = tf.keras.layers.add([x, y])
	#x = tf.keras.layers.Activation('relu')(x)
	return x

def residual_block_1(x, k, kernel_size, num_filters):
	y = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=kernel_size, 
		strides=1, 
		activation='relu')
	for i in range(k-1):
		activation = 'relu'
		if i == k-2:
			activation=None
		y = conv_block(
			x=y, 
			num_filters=num_filters, 
			kernel_size=1, 
			strides=1, 
			activation=activation)
	x = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=1, 
		strides=1, 
		activation='relu')
	x = tf.keras.layers.add([x, y])
	return x



def residual_block_with_pooling(x, k, kernel_size, num_filters):
	x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)(x)
	y = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=kernel_size, 
		strides=1, 
		activation='relu')
	for i in range(k-1):
		activation = 'relu'
		if i == k-2:
			activation=None
		y = conv_block(
			x=y, 
			num_filters=num_filters, 
			kernel_size=1, 
			strides=1, 
			activation=activation)
	x = conv_block(
		x=x, 
		num_filters=num_filters, 
		kernel_size=1, 
		strides=1, 
		activation='relu')
	x = tf.keras.layers.add([x, y])
	return x

def inverted_residual_block(inputs, stride,  num_filters, expansion):
	channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
	in_channels = tf.keras.backend.int_shape(inputs)[channel_axis]
	x = inputs
	x = conv_block(
		x=x, 
		num_filters=expansion * in_channels, 
		kernel_size=1, 
		strides=1, 
		activation='relu')
	x=depthwise_separable_conv(x=x, num_filters=num_filters, kernel_size=3, strides=stride, activation=None)
	if in_channels == num_filters and stride == 1:
		return tf.keras.layers.Add(name='add')([inputs, x])
	return x