'''
	evaluation.py
	contains the evaluation metrics for all the shit we're going to be doing
'''
	
import numpy as np
import pandas as pd
from data_prep import *

from sklearn.metrics import f1_score

def evaluate(predictions, labels, metric='f1'):
	'''
	'''

	if metric =='f1':
		return f1_score(predictions, labels, average="macro")