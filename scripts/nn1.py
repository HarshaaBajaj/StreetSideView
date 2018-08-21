'''
#***********************************************************
# DESCRIPTION 
# Main script to run precomplied models
	# Current configuration :
		# Model -- ResNet_18
		# Input -- (224,224,3)
	# Variations :
		# Model and layers -- Script can be modified to work with any precomplied model for any number of layers
		# Prediction technique -- Any classification algorithm can employed or the neural network can be used to make predictions
#***********************************************************
'''
# PROGRAM 
from utils1 import * # import helper functions

random.seed(73748932) ;
# create log and define it's name and format
log = logging.getLogger("neuralnets1")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')	

# define model and classifier to be used ---- changeable 
datapath =  '../data/' # data_rev
output = '../models/'
base_model_file =  'ResNet_18.model' # pre trained model
classifier =  'dnn'		#'svm': to keep pre-trained DNN fixed and use a SVM as classifier (this does not require a DNN, all other options do)
								#'dnn': to refine the DNN and use it as the classifier
								#'svmDnnRefined': this first refines the DNN (like option 'dnn') but then trains a SVM classifier on its output(like option 'svm')
name = base_model_file.split('.')[0]
model_results = output + '/'+name +'_' + classifier

if not os.path.exists(model_results):
    os.makedirs(model_results);
    print('Created',model_results)

#filename = '_'.join([name,classifier])

# create the logging file handler
fh = logging.FileHandler( model_results + '_'.join([name,classifier]) + ".log")
fh.setFormatter(formatter)
log.addHandler(fh) # add handler to log object
log.info("Program started")

labels = [0,1] 
train_list_path = datapath + 'train_list.txt'
valid_list_path = datapath +  'valid_list.txt'
test_list_path = datapath +  'test_list.txt'
	
#params
params = {
	'num_classes' : 2  , 
	'image_dimensions' : (224,224,3),
	
	'node' : "poolingLayer" if classifier=='svm' else [], # if svm then get feats from net (last global avg pooling layer) else get preds
	'mb_size' : 256, # mini batch size
	'dropout_rate' : 0.5 , # Droputout rate
	
	'freeze_weights' : False,   # Set to 'True' to freeze all but the very last layer. Otherwise the full network is refined
								#To "freeze" a layer means to exclude it from training, i.e. its weights will never be updated
	
	#gradient descent params
	'learn_rate' : [0.01] * 20 + [0.001] * 20 + [0.0001], # Learning rate schedule for step decay
	'beta_momentum_gd' : 0.9, # Momentum during gradient descent # beta - momentum decay for each mini batch
	'l2_reg_weight' : 0.0005, # L2 regularizer weight during gradient descent per sample
	'num_epochs' : 20,
	'epoch_size_train' : int(fsum(1 for line in open(train_list_path))),#10,
	'epoch_size_test' : int(fsum(1 for line in open(valid_list_path)))  #epoch_size_train = sum(len(v) for v in train_dict.values())
}
log.info('Initialized parameters')


# Create minibatches 
params['train_mbs'] = create_mb(train_list_path, params, True)
params['valid_mbs'] = create_mb(valid_list_path, params, False)
params['test_mbs'] = create_mb(test_list_path, params, False)
log.info('Created mini batches')


try :
    #model = load_model(model_results + filename + ".model")
	model= load_model(name + '_refined.model')
	log.info('Loaded existing model with layers :')
	node_outputs = get_node_outputs (model)
	[log.info('%s , %s' %(layer.name,layer.shape)) for layer in node_outputs]
	
except:
	# Define mapping from reader streams to network inputs
	label_input = input_variable(params['num_classes'])
	image_input = input_variable(params['image_dimensions'][::-1], name = "input")
	input_map = {
			image_input: params['train_mbs']['features'],
			label_input: params['train_mbs']['labels'] }
		
	# create model
	cntkModel = create_model(output+base_model_file, image_input, params)
	log.info('Created model with layers:')
	node_outputs = get_node_outputs (cntkModel)
	[log.info('%s , %s' %(layer.name,layer.shape)) for layer in node_outputs]
	
	# Instantiate the transfer learning model and loss function
	params['ce'] = cross_entropy_with_softmax(cntkModel, label_input)
	params['pe'] = classification_error(cntkModel, label_input) 
	
	# train model
	model = train_model(cntkModel,params,input_map)
	plt.savefig(model_results+'model_training_error.png',bbox_inches='tight', dpi = 200)
	
	#model.save(model_results + filename + ".model")
	model.save(model_results+name + '_refined.model')
	graph.plot(model, filename= model_results + name + "_refined_graph.pdf") # Write graph visualization
	log.info('Saved the model')

# Run net for all images

# run - node -- svm : pooling, dnn: []---- output=model
# if svm -- return feats else return pred

try:
	feats_train = pd.read_pickle(model_results + 'feats_train.pickle')
	labels_train = pd.read_pickle(model_results + 'labels_train.pickle')
	log.info('Loaded preobtained training features and labels')

except:
	feats_train,labels_train = runCntkModelAllImages(model, labels,train_list_path,params['train_mbs'], params['node'], params['mb_size'])
	#pickle.dump(dnnOutputTrain, open(model_results + 'dnnOutputTrain.pickle','wb')) ;
	pickle.dump(feats_train,open(model_results + 'feats_train.pickle','wb'))
	pickle.dump(labels_train,open(model_results + 'labels_train.pickle','wb'))
	log.info('Processed and saved new training features and labels')
try:
	feats_valid = pd.read_pickle(model_results + 'feats_valid.pickle')
	labels_valid = pd.read_pickle(model_results + 'labels_valid.pickle')
	log.info('Loaded preobtained testing features and labels')
except:
	feats_valid,labels_valid  = runCntkModelAllImages(model, labels,valid_list_path,params['valid_mbs'], params['node'], params['mb_size'])
	#pickle.dump(dnnOutputValid, open(model_results + 'dnnOutputValid.pickle','wb')) ;
	pickle.dump(feats_valid,open(model_results + 'feats_valid.pickle','wb'))
	pickle.dump(labels_valid,open(model_results + 'labels_valid.pickle','wb'))
	log.info('Processed and saved new validation features and labels')

try:
	feats_test = pd.read_pickle(model_results + 'feats_test.pickle')
	labels_test = pd.read_pickle(model_results + 'labels_test.pickle')
	log.info('Loaded preobtained testing features and labels')
except:
	feats_test,labels_test  = runCntkModelAllImages(model, labels,test_list_path,params['test_mbs'], params['node'], params['mb_size'])
	# if it is dnn classification, scores are returned not feats
	#pickle.dump(dnnOutputTest, open(model_results + 'dnnOutputTest.pickle','wb')) ;
	pickle.dump(feats_test,open(model_results + 'feats_test.pickle','wb'))
	pickle.dump(labels_test,open(model_results + 'labels_test.pickle','wb'))
	log.info('Processed and saved new testing features and labels')
	
FeatLabelInfo("Statistics training data:", feats_train, labels_train)	
FeatLabelInfo("Statistics validation data:", feats_valid, labels_valid)	
FeatLabelInfo("Statistics test data:", feats_test,  labels_test)

try:
	scoresMatrix = pd.read_pickle(model_results + 'scoresMatrix.pickle')
	log.info('Loaded preobtained scores')
except:
	if classifier.startswith('svm'):
	# Train SVMs for different values of C, and keep the best result
		try:
			bestLearner = np.load(model_results+'svm_bestlearner')
			log.info('Loaded preobtained best learner')
		except:
			(bestLearner, scoresMatrix) = train_SVM(feats_train,labels_train,feats_valid,labels_valid,feats_test,labels_test)
			#TOFO : fix training_accuracy_svm
			plt.savefig(model_results+'training_accuracy_svm.png',bbox_inches='tight', dpi = 200)
			pickle.dump(bestLearner, open(model_results + 'svm_bestlearner','wb')) #Store best model. Note that this should use a separate validation set, and not the test set.
			#np.load("d1.npy") ; print d1.get('key1')
	else : #DNN can be used directly as classifier
		scoresMatrix = np.vstack(feats_test)
	pickle.dump(scoresMatrix,open(model_results + 'scoresMatrix.pickle','wb'))

	
try:
	predLabels = pd.read_pickle(  model_results + "predLabels.pickle")
	log.info('Loaded preobtained predictions labels')
except:
	predLabels = [np.argmax(scores) for scores in scoresMatrix] # Predicted labels
	pickle.dump(  predLabels , open(model_results + "predLabels.pickle",   'wb'))
	log.info('Prediction labels generated and saved')


if not os.path.isfile(model_results + name + '_rocCurve_confMat.jpg') :
	classes = ['Likely House', 'Likely Land']
	fig = plt.figure(figsize=(14,6))
	plt.subplot(121)
	rocComputePlotCurves(labels_test, scoresMatrix, classes)
	# Plot confusion matrix
	# Note: Let C be the confusion matrix. Then C_{i, j} is the number of observations known to be in group i but predicted to be in group j.
	plt.subplot(122)
	confMatrix = confusion_matrix(labels_test, predLabels)
	cmPlot(confMatrix, classes, normalize=False)
	fig.savefig(model_results + name +'_roc_confMat.jpg', bbox_inches='tight', dpi = 200)
	log.info('Saved ROC and confusion matrix plots')

# Print accuracy to console
print('Classifier', classifier)
printAcc(confMatrix,classes,labels_test)

log.info('Done')