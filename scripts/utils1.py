
import os,random,pickle,datetime,logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cntk.io.transforms as xforms

from sklearn import svm
from scipy import interp
from cntk import load_model, Trainer, UnitType, use_default_device, placeholder, constant, cross_entropy_with_softmax, classification_error
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
from cntk.ops import input_variable, combine
from cntk.ops.functions import CloneMethod
from cntk.layers import placeholder, GlobalAveragePooling, Dropout, Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule,learning_parameter_schedule
from cntk.logging.graph import find_by_name
from cntk.logging import log_number_of_parameters, ProgressPrinter, graph
from cntk.logging import get_node_outputs

from itertools import *
from sklearn.metrics import *
from math import *

module_logger = logging.getLogger("neuralnets1.utils")
	
def create_mb_old(map_file, params, training_set):
	transforms = [] 
	image_dimensions = params['image_dimensions'] ;	
	num_classes = params['num_classes'] ;
	if training_set:
		# Scale to square-sized image. without this the cropping transform would chop the larger dimension of an
		# image to make it squared, and then take 0.9 crops from within the squared image.
		transforms += [xforms.scale(width=2*image_dimensions[0], height=2*image_dimensions[1], channels=image_dimensions[2], interpolations='linear', scale_mode='pad', pad_value=114)]
		transforms += [xforms.crop(crop_type='randomside', side_ratio=0.9, jitter_type='uniratio')]    # Randomly crop square area
						#randomside enables Horizontal flipping 
						#new_dim = side_ratio * min(old_w,old_h) , 0.9 * 224 = 201.6
		transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]
	# Scale down and pad
	transforms += [xforms.scale(width=image_dimensions[0], height=image_dimensions[1], channels=image_dimensions[2], interpolations='linear', scale_mode='pad', pad_value=114)]
		

	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
			features  = StreamDef(field='image', transforms=transforms),
			labels    = StreamDef(field='label', shape=num_classes))),
			randomize = training_set,
			multithreaded_deserializer=True)

def create_mb(map_file, params, training_set):
	transforms = [] 
	image_dimensions = params['image_dimensions'] ;
	num_classes = params['num_classes'] ;
	if training_set:
		# Scale to square-sized image. without this the cropping transform would chop the larger dimension of an
		# image to make it squared, and then take 0.9 crops from within the squared image.
		transforms += [xforms.scale(width=2*image_dimensions[0], height=2*image_dimensions[1], channels=image_dimensions[2],
									scale_mode='pad', pad_value=114)]
		transforms += [xforms.crop(crop_type='randomside', side_ratio=0.9, jitter_type='uniratio')]     # Randomly crop square area
						#randomside enables Horizontal flipping 
						#new_dim = side_ratio * min(old_w,old_h) , 0.9 * 224 = 201.6
		#transforms += [xforms.crop(crop_type='center')]
		transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]
	
	transforms += [xforms.crop(crop_type='center', side_ratio=0.875)] # test has no jitter]
	# Scale down and pad
	transforms += [xforms.scale(width=image_dimensions[0], height=image_dimensions[1], channels=image_dimensions[2], scale_mode='pad',
								pad_value=114)]

	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
			features  = StreamDef(field='image', transforms=transforms),
			labels    = StreamDef(field='label', shape=num_classes))),
			randomize = training_set,
			multithreaded_deserializer=True)
			
def create_model(base_model_file, input_features, params):
    num_classes = params['num_classes'];
    dropout_rate = params['dropout_rate'];
    freeze_weights = params['freeze_weights'] ; 
	
	# Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    log = logging.getLogger("neuralnets1.utils.create_model")
    log.info('Loaded base model - %s with layers:' % base_model_file)
    node_outputs = get_node_outputs ( base_model)
    [log.info('%s , %s' %(layer.name,layer.shape)) for layer in node_outputs]
    graph.plot(base_model, filename="base_model.pdf") # Write graph visualization
	
    feature_node = find_by_name(base_model, 'features')
    beforePooling_node = find_by_name(base_model, "z.x.x.r")
    
    # Clone model until right before the pooling layer, ie. until including z.x.x.r
    modelCloned = combine([beforePooling_node.owner]).clone(
        CloneMethod.freeze if freeze_weights else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Center the input around zero and set model input.
    # Do this early, to avoid CNTK bug with wrongly estimated layer shapes
    feat_norm = input_features - constant(114)
    model = modelCloned(feat_norm)

    # Add pool layer
    avgPool = GlobalAveragePooling(name = "poolingLayer")(model) # assign name to the layer and add to the model
	# Add drop out layer
    if dropout_rate > 0:
        avgPoolDrop = Dropout(dropout_rate)(avgPool) # add drop out layer with specified drop out rate and add it to the model
    else:
        avgPoolDrop = avgPool

    # Add new dense layer for class prediction
    finalModel = Dense(num_classes, activation=None, name="Dense") (avgPoolDrop)
    return finalModel

# Trains a transfer learning model
def train_model(cntkModel,params,input_map):
	log = logging.getLogger("neuralnets1.utils.train_model")
	mb_size = params['mb_size']
	num_epochs = params['num_epochs'] ;	
	epoch_size_train = params['epoch_size_train'] ; epoch_size_test = params['epoch_size_test'] ;
	minibatch_source_train = params['train_mbs'] ;
	minibatch_source_valid = params['valid_mbs'] ;
	#minibatch_source_test = params['test_mbs'] ;
	
	# Instantiate the trainer object
	#lr_schedule = learning_rate_schedule(params['learn_rate'], unit=UnitType.minibatch)
	lr_per_minibatch       = learning_parameter_schedule(params['learn_rate'],  minibatch_size=mb_size, epoch_size=epoch_size_train)
    
	mm_schedule = momentum_schedule(params['beta_momentum_gd'])
	learner = momentum_sgd(cntkModel.parameters, lr_per_minibatch, mm_schedule, l2_regularization_weight=params['l2_reg_weight'])
	progress_writers = [ProgressPrinter(tag='Training', num_epochs=num_epochs)]
	trainer = Trainer(cntkModel, (params['ce'], params['pe']), learner, progress_writers)
	
	# Run training epochs
	log.info('Training transfer learning model for %s epochs (epoch_size_train = %s ) .' % (num_epochs, epoch_size_train))
#   print("Training transfer learning model for {0} epochs (epoch_size_train = {1}).".format(num_epochs, epoch_size_train))
	errsVal  = []
	errsTrain = []
	log_number_of_parameters(cntkModel)
	
	for epoch in range(num_epochs):
		err_numer = 0
		sample_counts = 0
		while sample_counts < epoch_size_train:  # Loop over minibatches in the epoch
			sample_count = min(mb_size, epoch_size_train - sample_counts)
			data = minibatch_source_train.next_minibatch(sample_count, input_map = input_map)
			trainer.train_minibatch(data)        # Update model with it
			sample_counts += sample_count        # Count samples processed so far
			err_numer += trainer.previous_minibatch_evaluation_average * sample_count

			if sample_counts % (100 * mb_size) == 0:
				log.info ("Training: processed %s samples" %sample_counts)
		# Compute accuracy on training and test sets
		errsTrain.append(err_numer / float(sample_counts))
		trainer.summarize_training_progress()
		errsVal.append(cntkComputeTestError(trainer, minibatch_source_valid, mb_size, epoch_size_test, input_map))
		trainer.summarize_test_progress()

		# Plot training progress
		plt.plot(errsTrain, 'b-', errsVal, 'g-')
		plt.xlabel('Epoch number')
		plt.ylabel('Error')
		plt.title('Training error (blue), validation error (green)')
		plt.draw()
	return cntkModel
# Evaluate model accuracy
def cntkComputeTestError(trainer, minibatch_source_test, mb_size, epoch_size, input_map):
    acc_numer = 0
    sample_counts = 0
    while sample_counts < epoch_size:  # Loop over minibatches in the epoch
        sample_count = min(mb_size, epoch_size - sample_counts)
        data = minibatch_source_test.next_minibatch(sample_count, input_map = input_map)
        acc_numer     += trainer.test_minibatch(data) * sample_count
        sample_counts += sample_count
    return acc_numer / float(sample_counts)

def runCntkModelAllImages(model, classes, imgDir, mbs, node_name, mb_size = 1):
	log = logging.getLogger("neuralnets.utils.runCntkModelAllImages")
	# Create empty dnn output dictionary
	#dnnOutput = dict()
	#for label in classes:
	#	dnnOutput[label] = dict()

	imgPaths = [line.strip('\n') for line in open(imgDir)]# Prepare cntk input
	# Run CNTK model for each image
	num_classes = model.shape[0]
	image_dimensions = find_by_name(model, "input").shape[::-1]
	
	# Set output node
	if node_name == []:
		output_node = model # use final pred layer
	else:
		node_in_graph = model.find_by_name(node_name) # gives poolingLayer output 512*1*1
		output_node   = combine([node_in_graph.owner]) # Set output node
			
	# Evaluate DNN for all images
	feats = [] ; labels = [] ;  sample_counts = 0
	while sample_counts < len(imgPaths):
		sample_count = min(mb_size, len(imgPaths) - sample_counts)
		mb = mbs.next_minibatch(sample_count)
		output = output_node.eval(mb[mbs['features']])
		prev_count = sample_counts
		sample_counts += sample_count
		for path,o in zip(imgPaths[prev_count:sample_counts],output) :
			feat = o.flatten()
			feat /= np.linalg.norm(feat,2) #normalizing the features for this image
			(imgFilename, imgSubdir) = os.path.basename(path).split()
			#dnnOutput[int(imgSubdir)][imgFilename] = feat
			feats.append(np.float32(feat))
			labels.append(int(imgSubdir))

		if sample_counts % 100 < mb_size:
			log.info("Evaluating DNN (output dimension = %s) for image %s of %s: %s"%(len(feats[-1]), sample_counts, len(imgPaths), imgFilename))
	#repeat for train and test
	return feats,labels
	
def FeatLabelInfo(title, feats, labels, preString = "   "):
    log = logging.getLogger("neuralnets1.utils.printFeatLabelInfo"); log.info('%s' % title)
    log.info(preString + "Number of examples: %s" %(len(feats)))
    log.info(preString + "Number of positive examples: %s" % (sum(np.array(labels) == 1)))
    log.info(preString + "Number of negative examples: %s" % (sum(np.array(labels) == 0)))
    log.info(preString + "Dimension of each example: %s" % (len(feats[0])))

# SVM training params (script: 4_trainSVM.py)
def train_SVM(feats_train,labels_train,feats_valid,labels_valid,feats_test,labels_test):
	log = logging.getLogger("neuralnets1.utils.train_SVM");
	svm_CVals = [10**-4, 10**-3, 10**-2, 0.1, 1, 10, 100] # Slack penality parameter C to try during SVM training
	bestAcc = float('-inf')
	valAccs = []
	for svm_CVal in svm_CVals:
		log.info("Start SVM training  with C = %s" % format(svm_CVal))
		tstart = datetime.datetime.now()
		learner = svm.LinearSVC(C=svm_CVal, class_weight='balanced', verbose=0)
		learner.fit(feats_train, labels_train)
		log.info("Training time in seconds: %s" % ((datetime.datetime.now() - tstart).total_seconds() * 1000))
		log.info("Training accuracy    =  %s (percent)" % (100 * np.mean(sklearnAccuracy(learner, feats_train, labels_train))))
		valAcc = np.mean(sklearnAccuracy(learner, feats_valid,  labels_valid))
		log.info("Validation accuracy        =  %s (percent)" % (100 * valAcc))
		valAccs.append(valAcc)
		if valAcc > bestAcc:
			log.info("   ** Updating best model. **")
			bestC = svm_CVal ; bestAcc = valAcc ; bestLearner = learner
		plt.plot(svm_CVal, 'b-', valAcc, 'g-')
		plt.xlabel('Regularization Rate')
		plt.ylabel('Accuracy')
		plt.title('Regularization Rate (blue), Accuracy(green)')
		plt.draw()
	log.info("Best model has validation accuracy %s (percent) , at C = %s" % (100 * bestAcc, bestC))
	scoresMatrix = bestLearner.decision_function(feats_test)
	# If binary classification problem then manually create 2nd column
	# Note: scoresMatrix is of size nrImages x nrClasses
	if len(scoresMatrix.shape) == 1:
		scoresMatrix = [[-scoresMatrix[i],scoresMatrix[i]] for i in range(len(scoresMatrix))]
		scoresMatrix = np.array(scoresMatrix)
	return bestLearner,scoresMatrix	
	
def rocComputePlotCurves(gtLabels, scoresMatrix, labels):
    #Code taken from Microsoft AML Workbench iris tutorial
    n_classes = len(labels)
    Y_score = scoresMatrix
    Y_onehot = []
    for i in range(len(gtLabels)):
        Y_onehot.append([])
        for j in range(len(labels)):
            Y_onehot[i].append(0)

        Y_onehot[i][gtLabels[i]] = 1
    Y_onehot = np.asarray(Y_onehot)

    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], thres[i] = roc_curve(Y_onehot[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_onehot.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # fig = plt.figure(figsize=(6, 5), dpi=75)
    # set lineweight
    lw = 2

    # plot micro average
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plot macro average
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # plot ROC for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels[i], roc_auc[i]))

    # plot diagnal line
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    return (fpr, tpr, thres)
	
def sklearnAccuracy(learner, feats, gtLabels):
    estimatedLabels = learner.predict(feats)
    confusionMatrix = confusion_matrix(gtLabels, estimatedLabels)
    return cmGetAccuracies(confusionMatrix, gtLabels)
	
def cmSanityCheck(confMatrix, gtLabels):
    for i in range(max(gtLabels)+1):
        assert(sum(confMatrix[i,:]) == sum([l == i for l in gtLabels])) 

def cmGetAccuracies(confMatrix, gtLabels = []):
    if gtLabels != []:
        cmSanityCheck(confMatrix, gtLabels)
    return [float(confMatrix[i, i]) / sum(confMatrix[i,:]) for i in range(confMatrix.shape[1])]
		
def cmPlot(confMatrix, classes, normalize=False, title='Confusion matrix', cmap=[]):
    if normalize:
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        confMatrix = np.round(confMatrix * 100,1)
    if cmap == []:
        cmap = plt.cm.Blues

    #Actual plotting of the values
    thresh = confMatrix.max() / 2.
    for i, j in product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
        plt.text(j, i, confMatrix[i, j], horizontalalignment="center",
                 color="white" if confMatrix[i, j] > thresh else "black")

    avgAcc = np.mean([float(confMatrix[i, i]) / sum(confMatrix[:, i]) for i in range(confMatrix.shape[1])])
    plt.imshow(confMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title + " (avgAcc={:2.2f}%)".format(100*avgAcc))
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def printAcc(confMatrix,classes,labels_test):
    columnWidth = max([len(s) for s in classes])
    accs = [float(confMatrix[i, i]) / sum(confMatrix[i,:]) for i in range(confMatrix.shape[1])]
    [print(("Class {:<" + str(columnWidth) + "} accuracy: {:2.2f}%.").format(cls, 100 * acc)) for cls, acc in zip(classes, accs)]
    globalAcc = 100.0 * sum(np.diag(confMatrix)) / sum(sum(confMatrix))
    print("OVERALL accuracy: {:2.2f}%.".format(globalAcc))
    print("OVERALL class-averaged accuracy: {:2.2f}%.".format(100 * np.mean(accs)))
    
    
	
