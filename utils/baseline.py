'''
	baseline.py

	includes the baseline, oracle classes
'''

import numpy as np
import pandas as pd
from data_prep import *
from sklearn import neighbors
from sklearn.model_selection import KFold
from evaluation import *

class BaselineClassifier():


	def __init__(self, base_path):
		self.base_path = base_path

	def train(self, train_set, train_labels):
		pass


	def predict(self, test_set):
		'''
			should return a ndarray of shape (len(test_set), 28) representing predictions for each iamge in 28 classes 

		'''
		predictions = np.zeros(shape=(len(test_set), 28))
		for i, image_id in enumerate(test_set):
			print 'image #:', i, '\r',
			stack = load_image(self.base_path, image_id, factor=.25)
			match = self._dist_min(stack[0,:,:], stack[1:,:,:])

			indices = None
			# resembles microtubule reference, turn on 11, 14, 15, 17, 18, 19, 
			if match == 0: 
				indices = [11, 14, 15, 17, 18, 19]
				
			# resembles nucleus reference,  turn on 0, 2, 4, 5
			if match == 1: 
				indices = [0, 2, 4, 5]

			# resembles ER reference, turn on 6
			if match == 2: 
				indices = [6]

			predictions[i, indices] = 1


		print ""

		return predictions


	def _dist_min(self, target, references):
		'''
			get the index of the reference that minimizes sum of squared distance to target
		'''

		# min_score = float('inf')
		# min_idx = -1
		# for i, reference in enumerate(references):
		# 	score = np.sum(np.subtract(target, reference))
		# 	if score < min_score: 
		# 		min_score = score
		# 		min_idx = i
		# return i

		subtracted = np.zeros((3, target.shape[0], target.shape[1]))
		for i, reference in enumerate(references):
			subtracted[i] = np.subtract(reference, target)
		return np.argmin(np.sum(np.square(subtracted), axis=(1, 2)))



class OracleClassifier():


	def __init__(self):
		self.n_neighbors = 5


	def train(self, train_set, train_labels):

		pass
		neighbors = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors)

	def predict(self, test_set):
		pass



if __name__ == "__main__": 
	print 'testing the baseline'

	imagedir = "data/train/"
	
	NoI, _ = gather_image_set(imagedir)

	labels_csv = "data/train.csv"
	df, image_ids = read_labels(labels_csv)
	write_to_file("train_image_ids", image_ids)

	labels = get_labels(df, image_ids)
	write_to_file("train_labels", labels)

	n_splits = 10
	kf = KFold(n_splits=n_splits, shuffle=True)
	f1_scores = np.zeros(n_splits)
	
	for i, (train_index, test_index) in enumerate(kf.split(image_ids)):
		print 'Current processing fold {}'.format(i)

		train_image_ids = [image_ids[idx] for idx in train_index]
		train_image_labels = labels[train_index]
		test_image_ids = [image_ids[idx] for idx in test_index]
		test_image_labels = labels[test_index]

		baseline = BaselineClassifier(imagedir)
		baseline.train(train_image_ids, train_image_labels)
		predictions = baseline.predict(test_image_ids)

		score = evaluate(predictions, test_image_labels)
		f1_scores[i] = score

		print 'for fold {}, score is {}'.format(i, score)

	print f1_scores
	write_to_file("f1_scores_bl", f1_scores)




