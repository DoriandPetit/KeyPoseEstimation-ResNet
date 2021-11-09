from generator.stickman import stick_man_generator
from generator.test import generate_test_dataset
from generator.evaluation_dataset import load_validation
from model.DNN import create_DNN
from model.backbones import possible_backbones
from model.heads import possible_heads
from train.loop import training_loop
from train.metrics import metric_MPJPE, metric_3DPCK
from train.transfo import transform_labels_heatmaps, transform_labels_scalars
from save.saver import save_trained_model
from save.collect import get_trained_DNN
from visu.show import visu_eval_DNN
from evaluation_test.test import eval_DNN, evaluate_on_a_val_set, evaluate_on_a_val_set

import tensorflow as tf
import os
import argparse
#import nvidia_smi
import numpy as np
import sys

"""
def check(gpu):
	if gpu >= 0:
		nvidia_smi.nvmlInit()
		deviceCount = nvidia_smi.nvmlDeviceGetCount()
		gpus=[]
		for i in range(deviceCount):
			handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
			mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
			percentage_used = 100 * (mem_res.used / mem_res.total)
			gpus.append(percentage_used)
		gpu = np.argmin(gpus)
		if gpus[gpu] < 50:
			return gpu
		else:
			print('No GPU availalble (all used with at least 50% memory)')
			sys.exit()
	return gpu
"""

parser = argparse.ArgumentParser(
	usage=__doc__,
	formatter_class=argparse.RawDescriptionHelpFormatter)
	
parser.add_argument('-shape', nargs='?', type=int, default=196)
parser.add_argument('-head', choices=possible_heads, nargs='?', type=str, default='scalar')
parser.add_argument('-backbone', choices=possible_backbones, nargs='?', type=str, default='VGG')
parser.add_argument('-metric', choices=['MPJPE', '3DPCK'], nargs='?', type=str, default='MPJPE')
parser.add_argument('-gpu', nargs='?', type=int, default=-1)
parser.add_argument('-epochs', nargs='?', type=int, default=50)
parser.add_argument('-batch_size', nargs='?', type=int, default=64)
parser.add_argument('-lr', nargs='?', type=float, default=0.001)
parser.add_argument('-circles', nargs='?', type=float, default=0.5)
parser.add_argument('-squares', nargs='?', type=float, default=0.3)
parser.add_argument('-real', nargs='?', type=float, default=0.9)
parser.add_argument('-load', nargs='?', type=bool, default=False)
args = parser.parse_args()

"""
args.gpu=check(gpu=args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.gpu >= 0:
	try:
		gpus = tf.config.experimental.list_physical_devices('GPU')
		gpu = gpus[0]
		tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)
"""

val_set = stick_man_generator(
	batch_size = args.batch_size, 
	set_of_data = 'val', 
	p_circles=args.circles, 
	p_squares=args.squares, 
	p_real=args.real, 
	input_shape = (args.shape,args.shape,3))

if not args.load:
	val_set = stick_man_generator(
		batch_size = args.batch_size, 
		set_of_data = 'val', 
		p_circles=args.circles, 
		p_squares=args.squares, 
		p_real=args.real, 
		input_shape = (args.shape,args.shape,3))
	train_set = stick_man_generator(
		batch_size = args.batch_size, 
		set_of_data = 'train', 
		p_circles=args.circles, 
		p_squares=args.squares, 
		p_real=args.real, 
		input_shape = (args.shape,args.shape,3))

	DNN = create_DNN(
		input_shape=(args.shape,args.shape,3), 
		backbone=args.backbone, 
		head=args.head, 
		num_key_points=int(train_set.key_points.labels.shape[-1]/2))

	DNN.summary()

	metrics = {
		'MPJPE' : metric_MPJPE,
		'3DPCK' : metric_3DPCK,
	}
	transfo = {
		'heatmap' : transform_labels_heatmaps,
		'scalar' : transform_labels_scalars,
	}

	DNN=training_loop(
		head=args.head,
		DNN=DNN, 
		learning_rate=args.lr,
		epochs=args.epochs, 
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
		loss_fn=tf.keras.losses.MeanSquaredError(), 
		train_set=train_set, 
		val_set=val_set, 
		transform_labels=transfo[args.head], 
		metric=metrics[args.metric])
	image_shape = args.shape
else:
	DNN, dnn_config = get_trained_DNN(num_key_points=int(val_set.key_points.labels.shape[-1]/2))
	val_set = stick_man_generator(
		batch_size = args.batch_size, 
		set_of_data = 'val', 
		p_circles=args.circles, 
		p_squares=args.squares, 
		p_real=args.real, 
		input_shape = (dnn_config['shape'],dnn_config['shape'],3))
	image_shape = dnn_config['shape']

if not args.load:
	errors = eval_DNN(DNN=DNN, image_shape=image_shape)
	
	X_val, y_val = load_validation()
	val_data = [X_val,y_val]

	val_err = evaluate_on_a_val_set(DNN=DNN,image_shape=image_shape,eval_set=val_data)
	save_trained_model(parser=args, DNN=DNN, error = errors,val_error=np.round(val_err,5))
else:
	error = visu_eval_DNN(DNN=DNN, image_shape=image_shape)