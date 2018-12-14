"""
initialization.py

some of the functions that we can use to initialize our models, including loading from previous states or performing Xavier initialization.
"""
import os
import sys
from models import * 
from data_prep import * 
from evaluation import * 
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def initialize(options): 
	'''
	input: 
		* options, a dictionary containing all the paths and stuff that we're going to use to set up our training

	output:
		* model 		-- the model 
		* optimizer		-- optimizer that we're going to use
		* loss function -- loss function 
	
	'''

	# define model
	if options['model'] == 'simple_CNN':
		model = simple_CNN()
		model.apply(init_weights)

	elif options['model'] == 'v2': 
		model = larger_architecture()
		model.apply(init_weights)

	if options['model_continue'] is not None:
		model.load_state_dict(torch.load(options['model_continue']))

	# define optimizer
	if options['optimizer'] == 'adam': 
		optimizer = optim.Adam(model.parameters(), lr=.001, betas=(.9, .99))
	if options['optim_continue'] is not None:
		optimizer.load_state_dict(torch.load(options['optim_continue']))

	# define loss
	if options['loss'] == 'FocalLoss':
		loss = FocalLoss()

	return model, optimizer, loss