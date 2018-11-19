"""
	initialization.py

"""
import os
import sys
from models import * 
from data_prep import * 
from evaluation import * 
import torch

def initialize(options): 
	'''
	input: 
		* options, a dict containing options 

	output: 
		* model 
		* optimizer
		* loss function 
	'''

	# define model
	if options['model'] == 'simple_CNN':
		model = simple_CNN()
		model.apply(init_weights)
	if options['model_continue'] is not None:
		model = torch.load(options['model_continue'])

	# define optimizer
	if options['optimizer'] == 'adam': 
		optimizer = optim.Adam(model.parameters(), lr=.001, betas=(.9, .99))
	if options['optim_continue'] is not None:
		optimizer.load_state_dict(torch.load(options['optim_continue']))

	# define loss
	if options['loss'] == 'FocalLoss':
		loss = FocalLoss()


	return model, optimizer, loss