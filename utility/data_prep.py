'''
	data_prep.py
'''

import pandas as pd
import numpy as np
import subprocess
import pickle
import os

from skimage import io
from skimage.transform import rescale, resize

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

class HumanProteinDataset(Dataset):
    '''
    '''
    def __init__(self, labels_csv, root_dir, transform=None):
        self.label_names = {
            0:  "Nucleoplasm",  
            1:  "Nuclear membrane",   
            2:  "Nucleoli",   
            3:  "Nucleoli fibrillar center",   
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",   
            6:  "Endoplasmic reticulum",   
            7:  "Golgi apparatus",   
            8:  "Peroxisomes",   
            9:  "Endosomes",   
            10:  "Lysosomes",   
            11:  "Intermediate filaments",   
            12:  "Actin filaments",   
            13:  "Focal adhesion sites",   
            14:  "Microtubules",   
            15:  "Microtubule ends",   
            16:  "Cytokinetic bridge",   
            17:  "Mitotic spindle",   
            18:  "Microtubule organizing center",   
            19:  "Centrosome",   
            20:  "Lipid droplets",   
            21:  "Plasma membrane",   
            22:  "Cell junctions",   
            23:  "Mitochondria",   
            24:  "Aggresome",   
            25:  "Cytosol",   
            26:  "Cytoplasmic bodies",   
            27:  "Rods & rings"
        }

        self.labels_df = pd.read_csv(labels_csv)
        for _, row in self.labels_df.iterrows():
            labels = np.array(row.Target.split(" ")).astype(np.int)
            row.Target = np.array([1 if i in labels else 0 for i in range(28)])
            
        self.root_dir = root_dir
        self.transform = transform
        
        self.raw_h = 512
        self.raw_w = 512
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        '''
        
        '''
#        image_base = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image_stack = self._load_image(self.labels_df.iloc[idx, 0])
        
        sample = {'stack': image_stack, 'labels': self.labels_df['Target'].iloc[idx]}
        if self.transform: 
            sample = self.transform(sample)
            
        return sample
    
    def _load_image(self, image_id, factor = 1):
        image_stack = np.zeros((4,self.raw_w,self.raw_h))
        image_stack[0,:,:] = io.imread(self.root_dir + image_id + "_green" + ".png")
        image_stack[1,:,:] = io.imread(self.root_dir + image_id + "_red" + ".png")
        image_stack[2,:,:] = io.imread(self.root_dir + image_id + "_blue" + ".png")
        image_stack[3,:,:] = io.imread(self.root_dir + image_id + "_yellow" + ".png")

        if factor != 1:
            image_scaled = np.zeros(shape=(4, int(self.raw_w*factor), int(self.raw_h*factor)))
            image_scaled[0,:,:] = rescale(images[0,:,:], factor)
            image_scaled[1,:,:] = rescale(images[1,:,:], factor)
            image_scaled[2,:,:] = rescale(images[2,:,:], factor)
            image_scaled[3,:,:] = rescale(images[3,:,:], factor)
            return image_scaled

        return image_stack 
    

class Rescale(object):
    '''
    '''
    
    def __init__(self, scaled_dims):
        self.scaled_dims = scaled_dims
    
    def __call__(self, sample):
        stack_raw = sample['stack']
        
        stack_scaled = np.zeros(shape = (4, self.scaled_dims[0], self.scaled_dims[1]))
        stack_scaled[0,:,:] = resize(stack_raw[0, :, :], self.scaled_dims)
        stack_scaled[1,:,:] = resize(stack_raw[1, :, :], self.scaled_dims)
        stack_scaled[2,:,:] = resize(stack_raw[2, :, :], self.scaled_dims)
        stack_scaled[3,:,:] = resize(stack_raw[3, :, :], self.scaled_dims)
        
        return {'stack': stack_scaled, 'labels':sample['labels']}


class ToTensor(object):
    def __call__(self, sample):
        temp = sample['stack']/255.0
        totensor = transforms.ToTensor()
        sample['stack'] = totensor(temp.transpose((1, 2, 0)))
        return sample


def get_data_loaders(labels_csv = './data/train.csv', root_dir = './data/train/', ds_from_pkl=False):
    '''
    
    '''
    if ds_from_pkl:
    	data = read_from_file('HumanProteinDataset')

    else: 
    	data = HumanProteinDataset(labels_csv, root_dir, transform=transforms.Compose([
                                                          Rescale((256, 256)),
                                                          ToTensor()
                         ]))
    	write_to_file('HumanProteinDataset', data)

    indices = np.arange(len(data))
    indices_train = np.random.choice(indices, size=int(.75*len(data)), replace=False)
    indices_test = list(set(indices) - set(indices_train))
    
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_test = SubsetRandomSampler(indices_test)
    
    dataloader_train = DataLoader(data, batch_size=10, sampler=sampler_train, num_workers=5)
    dataloader_test = DataLoader(data, batch_size=10, sampler=sampler_test, num_workers=5)
    
    return {'train': dataloader_train, 'val': dataloader_test}
    

def gather_image_set(imagedir):
	'''
	Use a subprocess to ls into the provided image directory and list the content files. 
	More advanced features can be performed by adding regex to the subprocess, but it is not necessary with our 
	current organization. 

	'''
	   
	sp = subprocess.Popen('ls ' + imagedir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	image_files = [name.strip() for name in sp.stdout.readlines()]
	return (len(image_files), image_files)


def write_log(model_save_loc, optim_save_loc, metrics):
	'''
	'''

	print 'writing log '
	


def write_to_file(name, obj):
	'''
		Write object of specified name to a pickle file 
	'''

	print 'writing structures to pickle'
	print '----------------------------'

	path = os.getcwd() + '/pickles/' + name + '.pkl'
	file = open(path, 'wb')
	pickle.dump(obj, file)
	file.close()


def read_from_file(name):
	'''
		Return loaded object given by input name
	'''
	print 'reading structures from pickle'
	print '------------------------------'

	path = os.getcwd() + '/pickles/' + name + '.pkl'
	file = open(path, 'rb')
	new_obj = pickle.load(file)
	file.close()

	return new_obj


def load_image(basepath, image_id, factor=1):
	'''

	'''
	#print 'loading image with image_id', image_id, '\r'
	images = np.zeros((4,512,512))
	images[0,:,:] = io.imread(basepath + image_id + "_green" + ".png")
	images[1,:,:] = io.imread(basepath + image_id + "_red" + ".png")
	images[2,:,:] = io.imread(basepath + image_id + "_blue" + ".png")
	images[3,:,:] = io.imread(basepath + image_id + "_yellow" + ".png")

	if factor != 1:
		image_scaled = np.zeros(shape=(4, int(512*factor), int(512*factor)))
		image_scaled[0,:,:] = rescale(images[0,:,:], factor)
		image_scaled[1,:,:] = rescale(images[1,:,:], factor)
		image_scaled[2,:,:] = rescale(images[2,:,:], factor)
		image_scaled[3,:,:] = rescale(images[3,:,:], factor)
		return image_scaled

	return images


def fill_targets(row, label_names):
	'''

	'''

	row.Target = np.array(row.Target.split(" ")).astype(np.int)
	for num in row.Target:
		name = label_names[int(num)]
		row.loc[name] = 1
	return row


def read_labels(labels_csv):
	'''

	'''
	print 'reading the label csv'

	label_names = {
	    0:  "Nucleoplasm",  
	    1:  "Nuclear membrane",   
	    2:  "Nucleoli",   
	    3:  "Nucleoli fibrillar center",   
	    4:  "Nuclear speckles",
	    5:  "Nuclear bodies",   
	    6:  "Endoplasmic reticulum",   
	    7:  "Golgi apparatus",   
	    8:  "Peroxisomes",   
	    9:  "Endosomes",   
	    10:  "Lysosomes",   
	    11:  "Intermediate filaments",   
	    12:  "Actin filaments",   
	    13:  "Focal adhesion sites",   
	    14:  "Microtubules",   
	    15:  "Microtubule ends",   
	    16:  "Cytokinetic bridge",   
	    17:  "Mitotic spindle",   
	    18:  "Microtubule organizing center",   
	    19:  "Centrosome",   
	    20:  "Lipid droplets",   
	    21:  "Plasma membrane",   
	    22:  "Cell junctions",   
	    23:  "Mitochondria",   
	    24:  "Aggresome",   
	    25:  "Cytosol",   
	    26:  "Cytoplasmic bodies",   
	    27:  "Rods & rings"
	}

	df = pd.read_csv(labels_csv)

	for key in label_names.keys():
		df[label_names[key]] = 0

	df = df.apply(fill_targets, axis=1, args=((label_names)))

	return (df, df['Id'])


def get_labels(df, id_set):
	'''
	'''

	print 'extracting labels'
	new_df = df[df['Id'].isin(id_set)].copy()
	new_df.drop(['Id', 'Target'], axis=1, inplace=True)
	return new_df.as_matrix()


def partition_data(full_set):
	pass


if __name__ == "__main__":
	# print 'preparing dataset'
	# imagedir = "data/train/"
	# NoI, file_names = gather_image_set(imagedir)

	# labels_csv = "data/train.csv"
	# df, image_ids = read_labels(labels_csv)

	loaders = get_data_loaders()
	train_loader = loaders['train']
	print train_loader
	for x in train_loader:
		for i, j in x.iteritems():
			print i
			print j

