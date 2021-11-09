import os
import json
import numpy as np

from model.DNN import create_DNN

def get_trained_DNN(num_key_points):
	with open('model_registry.json') as json_file:
		data = json.load(json_file)
	print(data)
	columns = {'ID' : []}
	for key in list(data[list(data.keys())[0]].keys()):
		columns[key] = []
	string_sizes = np.zeros(2+len(list(data[list(data.keys())[0]].keys())))
	for key in list(data.keys()):
		if len(key) > string_sizes[0]:
			string_sizes[0] = len(key)
		for cpt,k in enumerate(list(data[key].keys())):
			elem = str(data[key][k])
			#print(elem)
			if string_sizes[1+cpt] < len(k):
				string_sizes[1+cpt] = len(k)
			if string_sizes[1+cpt] < len(elem):
				string_sizes[1+cpt] = len(elem)
	print('+' + '-' * int(np.sum(string_sizes) + 3 * len(string_sizes) - 1) + '+')
	print('|',end='')
	for cpt,key in enumerate(list(columns.keys())):
		print(' ' + key + ' '*int(1+ string_sizes[cpt] - len(key)), end='')
		print('|', end='')
	print()
	print('+' + '-' * int(np.sum(string_sizes) + 3 * len(string_sizes) - 1) + '+')
	for key in list(data.keys()):
		print('|',end='')
		print(' ' + key + ' '*int(1+ string_sizes[0] - len(key)),end='')
		print('|',end='')
		for cpt,k in enumerate(list(data[key].keys())):
			print(' ' + str(data[key][k]) + ' '*int(1+ string_sizes[1+cpt] - len(str(data[key][k]))),end='')
			print('|',end='')
		print()
	print('+' + '-' * int(np.sum(string_sizes) + 3 * len(string_sizes) - 1) + '+')

	Id= input('\nSelect an ID : ')
	while Id not in list(data.keys()):
		Id= input('Select an ID : ')
	DNN = create_DNN(
		input_shape=(data[Id]['shape'],data[Id]['shape'],3), 
		backbone=data[Id]['backbone'], 
		head=data[Id]['head'], 
		num_key_points=num_key_points)
	DNN.load_weights('trained_DNNs/' + Id + '.h5')
	return DNN, data[Id]
