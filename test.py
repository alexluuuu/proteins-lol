"""
test.py

Multilabel classification --  evaluates a pre-trained model on the Kaggle test data and writes it out.


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
import torch.optim as optim
import torchvision

from models import *
from utility.evaluation import *
from utility.data_prep import *
from utility.initialization import *


def test(model, dataloader, thresholds = np.array([.5]*28), device="cpu"): 
	"""

	input: 
		* model 		-- a pre-trained model from teh lsat training cycle
		* dataloader 	-- a dataloader object which which we can bring in batches of data at a time for the forward pass
		* thresholds 	-- thresholds we apply against the outputs of the fully connected layer of the net
		* device 		-- optionally, the device that we're going to use. (cpu rip)
	"""
	since = time.time()

	predictions = {}
	model.eval()
	for i, batch in enumerate(dataloader): 
		print('batch {}\r'.format(i))
		inputs = batch['stack']
		image_ids = batch['id']
		inputs = inputs.to(device, dtype=torch.float)

		outputs = model(inputs)
		preds = np.array([out > thresholds for out in outputs.data.numpy()])

		for image_id, prediction in zip(image_ids, preds): 
			predictions[image_id] = prediction

	print('')

	time_elapsed = time.time() - since

	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	return predictions


def write_predictions(predictions_dictionary): 
	"""
		a function to write predictions out to file
	"""

	out = {}
	for image_id, prediction in predictions_dictionary.iteritems():
		
		nonbinarized = []

		for pos, val in enumerate(prediction): 
			if val: nonbinarized.append(pos)

		out[image_id] = str(sorted(nonbinarized, reverse=True))

	out_df = pd.DataFrame.from_dict(out, orient='index')
	out_df.columns = ['Id', 'Predicted']

	out_df.to_csv('prediction.csv')


def main(): 

	# set up some basic paths
	model_save_loc = "./utility/weights/larger_architecture_2018_11_29.pt"

	# get the model saved from the previous training cycle
	model = larger_architecture()
	model.load_state_dict(torch.load(model_save_loc))

	# get the dataloader
	dataloader = get_test_loader()

	# call the evaluation function on the unknown test data
	predictions = test(model, dataloader)

	# pickle in case you mess up writing out the prediction csv
	write_to_file("prediction_cache", predictions)

	# write out the prediction csv
	write_predictions(predictions)



if __name__ == "__main__":

	print('time to test :)')
	main()