'''
evaluation.py
contains the evaluation metrics for all the shit we're going to be doing
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
	    gamma (TYPE): Description
	
	'''
	def __init__(self, gamma=2):
		"""Summary
		
		Args:
		    gamma (int, optional): Description
		"""
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		

	def forward(self, pred, target):
		"""Summary
		
		Args:
		    pred (TYPE): Description
		    target (TYPE): Description
		
		Returns:
		    TYPE: Description
		
		Raises:
		    ValueError: Description
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
	Args:
	    predictions (TYPE): Description
	    labels (TYPE): Description
	    metric (str, optional): Description
	
	Returns:
	    TYPE: Description
	'''

	if metric =='f1':
		return f1_score(predictions, labels, average="macro")