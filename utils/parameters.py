'''
	Script with library and parameter information
	HB'18
'''

# Importing essential libraries
import numpy as np
import glob as g
import os,time,shutil,imghdr,pickle,json,urllib,random,itertools
import cntk.io.transforms as xforms 
import pandas as pd
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from functools import reduce
from math import *
from sklearn.metrics import *
from itertools import *	
from cntk import *
from PIL import Image

from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
from cntk.logging import ProgressPrinter,log_number_of_parameters,graph,get_node_outputs
from cntk.layers import Convolution,BatchNormalization, GlobalAveragePooling, Dropout, Dense, MaxPooling,For,Sequential,\
						Convolution2D, Activation, default_options
from cntk.logging.graph import find_by_name,get_node_outputs

from cntk.learners import adadelta, learning_parameter_schedule_per_sample
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *

from cntk.train.distributed import *
from cntk.initializer import normal

import pandas as pd

# Set the directory paths
datapath = '../data/'
models = '../models'
if not os.path.exists(models): os.makedirs(models)

# To get the images	
labels = {'Likely House':0,'Likely Land':1} ; #'unknown' : 2 -- testing set
cols=['property_id','address_line1', 'address_city', 'state','indicator','assessor_photo'];

geolocator = Nominatim() ; 
key = '&key=AIzaSyAKjJpIt7WNkHd4q1VvJnmcewCODgZ4q40' ;
latlonurl = 'https://maps.googleapis.com/maps/api/geocode/json?address=' ;
imageurl = 'https://maps.googleapis.com/maps/api/streetview?size=640x480&location=';
staturl = 'https://maps.googleapis.com/maps/api/streetview/metadata?&location=';

# Path to reader
train_list_path = datapath + 'train_list.txt'
valid_list_path = datapath +  'valid_list.txt'
test_list_path = datapath +  'test_list.txt'

# Final classes
classes = ['Likely House', 'Likely Land']

# Changable model parameters
params = {
    'num_classes' : len(classes)  , 
    'image_dimensions' : (224,224,3),
    'mb_size' : 64, # mini batch size
    #gradient descent params
    'learn_rate' : [0.01] * 20 + [0.001] * 20 + [0.0001], # Learning rate schedule for step decay
    'beta_momentum_gd' : 0.9, # Momentum during gradient descent # beta - momentum decay for each mini batch
    'l2_reg_weight' : 0.0005, # L2 regularizer weight during gradient descent per sample
    'num_epochs' : 20
    #'epoch_size_train' : int(fsum(1 for line in open(train_list_path))),
    #'epoch_size_test' :  int(fsum(1 for line in open(valid_list_path)))
    }

'''
Debug notes
	lower mb_size to train the model faster
	change the learn_rate for zero loss error depending on the net
	decrease beta_momentum_gd to avoid getting stuck in local minima
	decrease l2_reg_weight to allow variation
	increase num_epochs for better results
	decrease epoch_size_test,epoch_size_train to get better results
'''