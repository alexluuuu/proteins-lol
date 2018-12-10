"""
saliency.py

"""
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models import * 
from utility.data_prep import * 
from utility.evaluation import *


def saliency(model, X, y): 
	"""
	Args:
		model (TYPE): Description
		X (TYPE): Description
		y (TYPE): Description
	"""
	model.eval()
	
	X_var = Variable(X, requires_grad=True)
	y_var = Variable(y)

	prediction = model(X_var)
	# print prediction
	# print y_var
	#prediction = prediction.gather(1, y_var.view(-1, 1)).squeeze()  

	# temp_grad = torch.FloatTensor(1, model_output.size()[-1]).zero_()
	# temp_grad[0][]

	#temp_grad = torch.from_numpy(y)

	prediction.backward(y_var.double())

	saliency = X_var.grad.data
	saliency = saliency.abs()
	saliency, i = torch.max(saliency,dim=1)
	saliency = saliency.squeeze() 
	
#	print saliency.shape
	return saliency.data


if __name__ == "__main__": 

	# define some paths
	labels_csv='./data/train.csv'
	root_dir='./data/train/'
	model_save_loc = "./utility/weights/larger_architecture_2018_11_29.pt"

	model = larger_architecture()
	model.load_state_dict(torch.load(model_save_loc))

#	data = HumanProteinDataset(labels_csv, root_dir)
	dataloaders = get_data_loaders(labels_csv=labels_csv, root_dir = root_dir)

	for i, batch in enumerate(dataloaders['val']): 
		# X = torch.from_numpy(batch['stack']).float()
		# y = torch.from_numpy(batch['labels']).float()
		X = batch['stack']
		y = batch['labels']
		idx = batch['idx']

		sal = saliency(model.double(), X, y)

		for j in range(sal.shape[0]):
			plt.imsave("test_%d_%d.png"%(i,j), sal[j,:,:])
			print idx


