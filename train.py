"""
	train.py


"""
from __future__ import print_function
from __future__ import division

import sys 
import time
import os
import copy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from models import *
from utility.evaluation import *
from utility.data_prep import *


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, thresholds = np.array([.5]*28), device="cpu"):
	'''
		train_model function adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

	'''
	since = time.time()

	val_F1_history = []
	val_loss_history = []
	train_loss_history = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_F1 = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for i, batch in enumerate(dataloaders[phase]):

				inputs = batch['stack']
				labels = batch['labels']

				inputs = inputs.to(device, dtype=torch.float)
				labels = labels.to(device, dtype=torch.float)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Get model outputs and calculate loss
					# Special case for inception because in training it has an auxiliary output. In train
					#   mode we calculate the loss by summing the final output and the auxiliary output
					#   but in testing we only consider the final output.
					if is_inception and phase == 'train':
						# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
#                        outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
#                        loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4*loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					preds = np.array([out > thresholds for out in outputs.data.numpy()])

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				print('Batch: {} | Epoch: {} | loss contribution: {}'.format(i, epoch, loss.item()*inputs.size(0)))
				# running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_F1 = np.mean(np.array(
				[ evaluate(pred, label) for pred, label in zip(preds, labels.data.numpy() ) ]
				)
			)

			print('{} Loss: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_F1))

			# deep copy the model
			if phase == 'val' and epoch_F1 > best_F1:
				best_F1 = epoch_F1
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == 'val':
				val_F1_history.append(epoch_F1)
				val_loss_history.append(epoch_loss)
			if phase == 'train':
				train_loss_history.append(epoch_loss)

		print()

	time_elapsed = time.time() - since

	# print some metrics
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val F1: {:4f}'.format(best_F1))

	# load best model weights
	model.load_state_dict(best_model_wts)

	return model, optimizer, {'val_F1': val_F1_history, 'val_loss': val_loss_history, 'train_loss': train_loss_history}


def run_training(model=None, optimizer=None, loss=None, labels_csv='./data/train.csv', root_dir='./data/train/', **kwargs):
	'''
	
	'''
	torch.cuda.empty_cache()
	if model is None: 
		assert model is not None, "model not provided"
		return 

	if optimizer is None: 
		assert optimizer is not None, "optimizer not provided"
		return 

	if loss is None: 
		assert loss is not None, "loss not provided"
		return

	dataloaders = get_data_loaders(labels_csv=labels_csv, root_dir = root_dir, ds_from_pkl = True)

	model, optimizer, metrics = train_model(model, 
		dataloaders, 
		#criterion=nn.BCEWithLogitsLoss(reduction='sum'),
		criterion=loss,
		optimizer=optimizer,
		num_epochs=num_epochs, 
		thresholds = np.array([.5]*28)
		)

	return model, optimizer, metrics


def main():

	options = {
		"model": "simple_CNN",
		"optimizer": "adam",
		"loss": "FocalLoss"
		"model_continue": "./utility/weights/simple_cnn_2018_11_17.pt",
		"optim_continue": None,
	}

	train_options = {
		"num_epochs": 20
	}

	# set up basic paths
	labels_csv='./data/train.csv'
	root_dir='./data/train/'
	model_save_loc = './utility/weights/simple_cnn_2018_11_17.pt'
	optim_save_loc = './utility/weights/adam_2018_11_19.pt'

	# set up the model, optimizer, loss based on options
	model, optimizer, loss = initialize(options)

	# train the model
	model, optimizer, metrics = run_training(model=model, 
											optimizer = optimizer, 
											loss = loss,
											labels_csv=labels_csv,
											root_dir = root_dir,
											train_options
											)

	# save the model
	torch.save(model.state_dict(), model_save_loc)

	# save the optimizer
	optimizer.save(optimizer.state_dict(), optim_save_loc)

	# write log 
	write_log(model_save_loc, optim_save_loc, metrics)



	print(history)



if __name__ == "__main__":

	print('time to train :)')
	main()
