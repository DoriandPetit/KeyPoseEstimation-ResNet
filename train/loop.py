import tensorflow as tf
import numpy as np
import time as time

learning_multiplier = 1

def convert_time(total_s):
	total_s = np.round(total_s / 60) * 60
	time_h = np.floor(total_s / 3600)
	total_s -= time_h * 3600
	time_m = np.floor(total_s / 60)
	total_s -= time_m * 60
	time_s = total_s
	return '%ih:%im:%.2fs'%(time_h,time_m,time_s)

def training_loop(DNN, head, epochs, optimizer, learning_rate, loss_fn, train_set, val_set, transform_labels, metric):
	average_time_step = []
	optimizer.lr = learning_rate
	global learning_multiplier
	if head == 'heatmap':
		learning_multiplier = 1000
	total_steps = epochs*train_set.__len__()
	for epoch in range(epochs):
		start_epoch = time.time()
		for step in range(train_set.__len__()):
			start_step = time.time()
			images, KP = train_set.__getitem__(index=step)
			with tf.GradientTape() as tape:
				logits = DNN(images, training=True)
				loss_value = loss_fn(transform_labels(KP, logits.shape[-2]), logits)
				loss_value += sum(DNN.losses)
				loss_value *= learning_multiplier
			grads = tape.gradient(loss_value, DNN.trainable_weights)
			optimizer.apply_gradients(zip(grads, DNN.trainable_weights))
			average_time_step.append(time.time()-start_step)
			print("\rEpoch %i/%i - Step %i/%i - Loss : %s%.3f (remaining time : %s)" 
				%(epoch+1, epochs, step+1, train_set.__len__(), ' '*(4-len(str(int(np.round(loss_value))))), loss_value, convert_time(
					total_s=(total_steps - epoch*train_set.__len__() - step) * np.mean(average_time_step))), end='')
	print()
	return DNN
