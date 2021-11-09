import numpy as np

def softmax2D_coords(heatmap):
	fmwidth=heatmap.shape[0]
	maxa=np.max(heatmap)
	soft_argmax_x=np.zeros((fmwidth,fmwidth))
	soft_argmax_y=np.zeros((fmwidth,fmwidth))
	for i in range(1,fmwidth+1,1):
		soft_argmax_x[i-1,:] = i / fmwidth
	for j in range(1,fmwidth+1,1):
		soft_argmax_y[:,j-1]=j / fmwidth
	array_softmax=np.exp(heatmap[:,:,0]-maxa)/np.sum(np.exp(heatmap[:,:,0]-maxa))
	xcoord=np.sum(np.multiply(array_softmax,soft_argmax_x))
	ycoord=np.sum(np.multiply(array_softmax,soft_argmax_y))
	return round(xcoord*5)-1,round(ycoord*5)-1

def max_coords(heatmap):
	maxa=np.max(heatmap)
	coorda=np.array(np.where(heatmap==maxa))[:,0]
	coorda=np.squeeze(coorda)
	return coorda[0], coorda[1]

def metric_MPJPE(pred, label):
	if len(pred.shape) == 5:
		score = []
		for b in range(len(pred)):
			for h in range(len(pred[b])):
				x,y=max_coords(heatmap=pred[b,h,:,:,:])
				norm = (label[b,::2][h] - x)**2 + (label[b,1::2][h] - y)**2
				score.append(np.sqrt(norm))
		score = np.mean(score)
	else:
		score = np.mean(np.sqrt((pred[:,::2] - label[:,::2])**2 + (pred[:,1::2] - label[:,1::2])**2))
	return score

def metric_3DPCK(pred, label):
	diameters = []
	for b in range(len(pred)):
		diameters.append(np.sqrt((np.max(label[b,::2]) - np.min(label[b,::2]))**2 + (np.max(label[b,1::2]) - np.min(label[b,1::2]))**2))
	diameters = 0.15*np.array(diameters)
	if len(pred.shape) == 5:
		num_correct = 0
		num_total = 0
		for b in range(len(pred)):
			for h in range(len(pred[b])):
				num_total += 1
				x,y=max_coords(heatmap=pred[b,h,:,:,:])
				if np.sqrt((label[b,::2][h] - x)**2 + (label[b,1::2][h] - y)**2) < diameters[b]:
					num_correct+=1
		score = num_correct/num_total
	else:
		num_correct = 0
		distances = np.sqrt((pred[:,::2] - label[:,::2])**2 + (pred[:,1::2] - label[:,1::2])**2)
		for b in range(len(distances)):
			num_correct += (distances[b,:] < diameters[b]).sum()
		score = num_correct / np.prod(distances.shape)
	return score