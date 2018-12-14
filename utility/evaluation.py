'''
evaluation.py
contains the evaluation metrics for all the shit we're going to be doing

implementation of the focal loss borrowed from
https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

'''
	
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from data_prep import *


class FocalLoss(nn.Module):
	'''
	Attributes:
	    gamma (float): parameter that is employed in focal loss computation
	
	'''
	def __init__(self, gamma=2):
		"""Summary
		
		Args:
		    gamma (int, optional): float that is used to compute the focal loss
		"""
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		

	def forward(self, pred, target):
		"""
			define as a forward pass because subclassing nn.Module
		"""
		if not (target.size() == pred.size()):
			raise ValueError("Target size ({}) must be the same as pred size ({})"
							 .format(target.size(), pred.size()))

		max_val = (-pred).clamp(min=0)
		loss = pred - pred * target + max_val + \
			((-max_val).exp() + (-pred - max_val).exp()).log()

		invprobs = nn.functional.logsigmoid(-pred * (target * 2.0 - 1.0))
		loss = (invprobs * self.gamma).exp() * loss
		
		return loss.sum(dim=1).mean()


def evaluate(predictions, labels, metric='f1'):
	'''
	input: 
		* predictions 		-- the prediction matrix w/ a 28-vector for each image in the batch
		* labels 			-- the correct 28 vector for each image in the batch
		* metric 			-- optionally, the metric which should be used to evaluate how well we did in predictions
		
	output: 
		* the score as determined by the evaluation metric

	'''

	if metric =='f1':
		return f1_score(predictions, labels, average="macro")

