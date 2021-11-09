import json
import numpy as np
import os
import sys

def save_trained_model(parser, DNN, error,val_error):
	if 'model_registry.json' in os.listdir():
		with open('model_registry.json') as json_file:
			data = json.load(json_file)
	else:
		data = {}
	Id=np.random.randint(low=0, high=10**8)
	data[Id] = {
		'shape' : parser.shape,
		'head' : parser.head,
		'backbone' : parser.backbone,
		'epochs' : parser.epochs,
		'batch_size' : parser.batch_size,
		'learning rate' : parser.lr,
		'circles' : parser.circles,
		'squares' : parser.squares,
		'real' : parser.real,
		'error val': val_error,
		'error1' : error[0],
		'error2' : error[1],
		'error3' : error[2],
		'error4' : error[3],
		
	}
	if 'trained_DNNs' not in os.listdir():
		os.mkdir('trained_DNNs')
	DNN.save_weights('trained_DNNs/' + str(Id) + '.h5')
	with open('model_registry.json', 'w') as outfile:
		json.dump(data, outfile)