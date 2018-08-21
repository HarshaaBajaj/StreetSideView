'''
	Script with essential helper functions
	HB'18
'''

from parameters import *

'''
	Model creation functions
'''
#ref : https://github.com/Microsoft/CNTK/blob/35255ed03bd0ffe6fdc536a4900e90bca5d38efe/Examples/Image/Classification/
# util funs
def convolution_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal(seed=938462), activation=relu):
    if activation is None:
        activation = lambda x: x
        
    r = Convolution(filter_size, 
                             num_filters, 
                             strides=strides, 
                             init=init, 
                             activation=None, 
                             pad=True, bias=False)(input)
    r = BatchNormalization(map_rank=1)(r)
    r = activation(r)
    
    return r

def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters)
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters, strides=(2,2))
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)

    s = convolution_bn(input, (1,1), num_filters, strides=(2,2), activation=None)
    
    p = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_filters, num_stack):
    assert (num_stack > 0)
    
    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r

def LocalResponseNormalization(k, n, alpha, beta, name=''):
    x = placeholder(name='lrn_arg')
    x2 = square(x)
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
    x2s = reshape(x2, (1, InferredDimension), 0, 1)
    W = constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = reshape(y, InferredDimension, 0, 2)
    den = exp(beta * log(k + b))
    apply_x = element_divide(x, den)
    return apply_x

# model funs
def create_feedforward(input,out_dims):
    num_hidden_layers = 1;  hidden_layers_dim = 200
    z = Sequential([For(range(num_hidden_layers), lambda i: Dense(hidden_layers_dim, activation=relu)),
                    Dense(out_dims)])(input)
    return z

def create_alexnet(input,out_dims):
	with default_options(activation=None, pad=True, bias=True):
	
	    z = Sequential([
			# we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
		    Convolution2D((11,11), 96, init=normal(0.01), pad=False, strides=(4,4), name='conv1'),
		    Activation(activation=relu, name='relu1'),
		    LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm1'),
		    MaxPooling((3,3), (2,2), name='pool1'),

		    Convolution2D((5,5), 192, init=normal(0.01), init_bias=0.1, name='conv2'), 
		    Activation(activation=relu, name='relu2'),
		    LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm2'),
		    MaxPooling((3,3), (2,2), name='pool2'),

		    Convolution2D((3,3), 384, init=normal(0.01), name='conv3'), 
		    Activation(activation=relu, name='relu3'),
		    Convolution2D((3,3), 384, init=normal(0.01), init_bias=0.1, name='conv4'), 
		    Activation(activation=relu, name='relu4'),
		    Convolution2D((3,3), 256, init=normal(0.01), init_bias=0.1, name='conv5'), 
		    Activation(activation=relu, name='relu5'), 
		    MaxPooling((3,3), (2,2), name='pool5'), 

		    Dense(4096, init=normal(0.005), init_bias=0.1, name='fc6'),
		    Activation(activation=relu, name='relu6'),
		    Dropout(0.5, name='drop6'),
		    Dense(4096, init=normal(0.005), init_bias=0.1, name='fc7'),
		    Activation(activation=relu, name='relu7'),
		    Dropout(0.5, name='drop7'),
		    Dense(out_dims, init=normal(0.01), name='fc8')
	    ])(input)
	return z

def create_vgg(input,out_dims):
	with default_options(activation=None, pad=True, bias=True):
		z = Sequential([
		# we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU)
		For(range(2), lambda i: [
			Convolution2D((3,3), 64, name='conv1_{}'.format(i)),
			Activation(activation=relu, name='relu1_{}'.format(i)),
		]),
		MaxPooling((2,2), (2,2), name='pool1'),

		For(range(2), lambda i: [
			Convolution2D((3,3), 128, name='conv2_{}'.format(i)),
			Activation(activation=relu, name='relu2_{}'.format(i)),
		]),
		MaxPooling((2,2), (2,2), name='pool2'),

		For(range(3), lambda i: [
			Convolution2D((3,3), 256, name='conv3_{}'.format(i)),
			Activation(activation=relu, name='relu3_{}'.format(i)),
		]),
		MaxPooling((2,2), (2,2), name='pool3'),

		For(range(3), lambda i: [
			Convolution2D((3,3), 512, name='conv4_{}'.format(i)),
			Activation(activation=relu, name='relu4_{}'.format(i)),
		]),
		MaxPooling((2,2), (2,2), name='pool4'),

		For(range(3), lambda i: [
			Convolution2D((3,3), 512, name='conv5_{}'.format(i)),
			Activation(activation=relu, name='relu5_{}'.format(i)),
		]),
		MaxPooling((2,2), (2,2), name='pool5'),

		Dense(4096, name='fc6'),
		Activation(activation=relu, name='relu6'),
		Dropout(0.5, name='drop6'),
		Dense(4096, name='fc7'),
		Activation(activation=relu, name='relu7'),
		Dropout(0.5, name='drop7'),
		Dense(out_dims, name='fc8')
	])(input)
	return z

def create_resnet(input, out_dims):
    C1 = convolution_bn(input, (7,7), 64,strides=(2,2)) # 112*64 
    pool1 = MaxPooling((3, 3), strides=(2, 2), pad=True)(C1) # 56*64
    
    C2 = resnet_basic_stack(pool1, 64, 2) # 56*64 

    C3_1 = resnet_basic_inc(C2, 128) # 28*128 
    C3_2 = resnet_basic_stack(C3_1, 128, 1) # 28*128 

    C4_1 = resnet_basic_inc(C3_2, 256) # 14*256
    C4_2 = resnet_basic_stack(C4_1, 256, 1) # 14*256
    
    C5_1 = resnet_basic_inc(C4_2,512) #7*512
    C5_2 = resnet_basic_stack(C5_1,512,1) # 7*512
    C5_3 = resnet_basic_stack(C5_1,512,1) # 7*512

    # Global average pooling
    pool = GlobalAveragePooling()(C5_2) # 1*512
    drop = Dropout(.25)(pool)
    net = Dense(out_dims, init=he_normal(seed=938462), activation=None, name = 'Dense')(drop)
    return net

'''
	Shared functions
'''

def create_mb(map_file, params, training_set):
	'''
		Create minibatches
		Input -- reader file, image dimensions & num classes , if training set
		Output -- Minibatches
	'''
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
		transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]

    # Scale down and pad
	transforms += [xforms.scale(width=image_dimensions[0], height=image_dimensions[1], channels=image_dimensions[2], scale_mode='pad',
								pad_value=114)]

	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
			features  = StreamDef(field='image', transforms=transforms),
			labels    = StreamDef(field='label', shape=num_classes))),
			randomize = training_set,
			multithreaded_deserializer=True)


def train_and_evaluate(params,model):
    '''
		Create, train and evaluate the model
		Input -- model parameters, type of model
		Output -- trained model
	'''
    num_classes = params['num_classes']
    train_mbs = params['train_mbs'] ; test_mbs = params['valid_mbs'] ;
    # training config
    num_epochs = params['num_epochs'] ;	
    epoch_size_train     = params['epoch_size_train']
    epoch_size_test     = params['epoch_size_test']
    minibatch_size = params['mb_size']

    # Input variables denoting the features and label data
    label_var = input_variable(num_classes)
    input_var = input_variable(params['image_dimensions'][::-1], name = "input")
    
    input_var_norm = input_var - constant(114) #z-score scaling #Center the input around zero
    
    z = model(input_var_norm, num_classes) # create model
   
    ## Training 

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)
    
    # Set training parameters
    lr_per_minibatch   = learning_parameter_schedule(params['learn_rate'],  minibatch_size=minibatch_size, epoch_size=epoch_size_train)
    momentums       = momentum_schedule(params['beta_momentum_gd'], minibatch_size = minibatch_size)
  
    # trainer object
    learner = momentum_sgd(z.parameters, lr = lr_per_minibatch,  momentum = momentums, l2_regularization_weight=params['l2_reg_weight'])
    progress_printer = ProgressPrinter(tag='Training', num_epochs=num_epochs)
    trainer = Trainer(z, (ce, pe), [learner], [progress_printer])
	
    # define mapping from reader streams to network inputs
    input_map = {
            z.arguments[0]: train_mbs['features'],
            label_var: train_mbs['labels'] }

    log_number_of_parameters(z) ;

    # start training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(num_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size_train:  # loop over minibatches in the epoch
            data = train_mbs.next_minibatch(min(minibatch_size, epoch_size_train - sample_count),input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)   # update model with it
            sample_count += data[label_var].num_samples   # count samples processed so far

            # visualization         
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()
	# evaluate the model 	
    test_err = evaluate(epoch_size_test,minibatch_size,test_mbs,input_map,trainer,label_var)

    # Visualize training result:
    window_width            = 32
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
    
    plt.figure(1)
    plt.subplot(121)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')
    plt.subplot(122)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    #plt.show()
    plt.draw()
	# return softmax of model
    return softmax(z,name = 'softmax')	

def eval_test_images(loaded_model, test_map_file, image_dims):
	'''
		Make predictions
		Input -- model, file reader, image dimensions
		Output -- predictions with probablities as data frame
	'''
	num_images =  int(fsum(1 for line in open(test_map_file)))
	print("Evaluating model for {0} images.".format(num_images))

	pred_count = 0 ; correct_count = 0
	predictions = pd.DataFrame(columns=['image_path', 'true_label', 'predicted_label','house_prob','land_prob'])
	with open(test_map_file, "r") as input_file:
		for line in input_file: # for each image
			tokens = line.rstrip().split('\t')
			img_file = tokens[0]

			try: 
				img = Image.open(img_file) # open image
				resized = img.resize(image_dims[:-1], Image.ANTIALIAS) # resize
				bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]] -114 # transform as per training
				hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
				arguments = {loaded_model.arguments[0] : hwc_format}
				probs = loaded_model.eval(arguments)[0] # get model predictions
			except FileNotFoundError:
				print("Could not open (skipping file): ", image_path)
				probs = ['None']
			
			true_label = int(tokens[1])
			predicted_label = np.argmax(probs) # find highest probablity class
			if predicted_label == true_label:
				correct_count += 1
				
			predictions.loc[pred_count] = [img_file,true_label,predicted_label,probs[0],probs[1]] # save info
			pred_count += 1
			if pred_count % 1000 == 0: # print status
				print("Processed {0} samples ({1:.2%} correct)".format(pred_count,(float(correct_count) / pred_count)))
			if pred_count >= num_images:
				break
			accuracy = (float(correct_count) / pred_count)
	print ("{0} of {1} prediction were correct with {2:.2f}% accuracy".format(correct_count, pred_count,accuracy))
	return predictions

def evaluate(epoch_size_test,minibatch_size,test_mbs,input_map,trainer,label_var):
	'''
		 Evaluating model with validation set
		 Input -- model, reader, image dimensions
		 Output -- Test error
	'''
	metric_numer    = 0 ;  metric_denom    = 0
	sample_count    = 0 ;   minibatch_index = 0

	while sample_count < epoch_size_test:
		current_minibatch = min(minibatch_size, epoch_size_test - sample_count) # get batch
		data = test_mbs.next_minibatch(current_minibatch, input_map=input_map)

		metric_numer += trainer.test_minibatch(data) * current_minibatch # test the model
		metric_denom += current_minibatch

		# Keep track of the number of samples processed so far.
		sample_count += data[label_var].num_samples
		minibatch_index += 1
	test_err = (metric_numer*100.0)/metric_denom # calculate test error
	print("---------------------------")
	print("Final Results: test errs = ",test_err)
	return test_err

def graphs(predictions):
	'''
		Draw the ROC and CM plots
		Input -- data frame with true and predicted labels
		Output -- plots
	'''
	cols= ['house_prob','land_prob']
	scoresMatrix=np.array(predictions[cols])
	plt.figure(figsize=(14,6))
	# ROC
	plt.subplot(121)
	rocComputePlotCurves(list(predictions.true_label), scoresMatrix, classes)
	# confusion matrix
	plt.subplot(122)
	cm = confusion_matrix(list(predictions.true_label), list(predictions.predicted_label))
	cmPlot(cm, classes) ; plt.draw()
	printAcc(cm,classes,list(predictions.true_label)) # print class level accuracy

def cmPlot(confMatrix, classes):
	'''
		Draw confusion matrix
		Input -- confusion matrix, class names
		Output -- confusion matrix plot
	'''
	cmap = plt.cm.Blues # setting colors -- darker --> stronger predictions
	#Actual plotting of the values
	thresh = confMatrix.max() / 2. # for color
	for i, j in product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
		plt.text(j, i, confMatrix[i, j], horizontalalignment="center", color="white" if confMatrix[i, j] > thresh else "black")

	avgAcc = fsum(np.diag(confMatrix)) / confMatrix.sum() # calculate accuracy
	plt.imshow(confMatrix, interpolation='nearest', cmap=cmap)
	plt.title("Confusion matrix (avgAcc={:2.2f}%)".format(100*avgAcc))
	plt.colorbar()
	plt.xticks(np.arange(len(classes)), classes, rotation=45)
	plt.yticks(np.arange(len(classes)), classes)
	plt.xlabel('Predicted label') ; plt.ylabel('True label')

def printAcc(confMatrix,classes,labels_test):
	'''
		Print class and final accuracies
		Input -- confusion matrix, class names, true labels
		Output -- Standard printed output
	'''
	columnWidth = max([len(s) for s in classes])
	accs = [float(confMatrix[i, i]) /  fsum(confMatrix[i,:]) for i in range(confMatrix.shape[1])] # class accuracies
	[print(("Class {:<" + str(columnWidth) + "} accuracy: {:2.2f}%.").format(cls, 100 * acc)) for cls, acc in zip(classes, accs)]
	globalAcc = 100.0 * fsum(np.diag(confMatrix)) / confMatrix.sum() # final accuracy
	print("OVERALL accuracy: {:2.2f}%.".format(globalAcc))
	print("OVERALL class-averaged accuracy: {:2.2f}%.".format(100 * np.mean(accs)))

def rocComputePlotCurves(true_labels, scoresMatrix, labels):
	'''
		Plot the Reciever Operating Curve
		Input -- true labels, pred probablities, class names
		Output -- True positive rate, false positive rate, threshold and ROC plot
	'''
	Y_onehot = []; n_classes = len(labels)
	Y_score = scoresMatrix
	for i in range(len(true_labels)):
		Y_onehot.append([])
		for j in range(len(labels)):
			Y_onehot[i].append(0)
		Y_onehot[i][true_labels[i]] = 1
	Y_onehot = np.asarray(Y_onehot)

	fpr = dict() ; tpr = dict() ; thres = dict() ; roc_auc = dict()

	for i in range(n_classes):
		fpr[i], tpr[i], thres[i] = roc_curve(Y_onehot[:, i], Y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	fpr["micro"], tpr["micro"], _ = roc_curve(Y_onehot.ravel(), Y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr ;  tpr["macro"] = mean_tpr ; roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# plot micro average
	plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),
			 color='deeppink', linestyle=':', linewidth=4)
	# plot macro average
	plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
			 color='navy', linestyle=':', linewidth=4)
	# plot ROC for each class
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=2,label='ROC curve of class {0} (area = {1:0.2f})'''.format(labels[i], roc_auc[i]))
	# plot diagnal line
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	return (fpr, tpr, thres)



	
def vis_images(paths):
	#land_house = list(map(lambda x  : ''.join(glob.glob(datapath + 'test/**/'+x)),pred[pred.predicted_label==0]['image_path'].sample(12)))
	#house_land = list(map(lambda x  : ''.join(glob.glob(datapath + 'test/**/'+x)),pred[pred.predicted_label==1]['image_path'].sample(12)))
	#print('Predicted as Likely House')
	#vis_images(land_house)
    n = len (paths)
    r = np.ceil(np.sqrt(n)) ; c = np.ceil(n/r) ;
    names = list(map(lambda x : os.path.basename(x).strip('.jpg'),paths)) 
    title = 'True Lable -' + paths[0].split('\\')[-2]  #'-'.join(paths[0].split('/')[2:5])
    fig = plt.figure(figsize=figsiz)
    for i in range(1,n+1):
        ax = fig.add_subplot(r,c,i)
        ax.imshow(plt.imread(paths[i-1]));
        ax.set_title(names[i-1],fontsize = 15)
        ax.axis('off')
    fig.set_tight_layout(True)
    fig.suptitle(title,fontsize = 18, color= 'red', y=1.02)
	
#ensemble
def ensemble(df1,df2,w=[.5,.5]):
	#w = np.array([.6,.4])
	#new = ensemble(predictions,predictions1,w)
	assert(len(df1==df2))
	assert(fsum(w)==1)
	new = df1.merge(df2,on=['image_path','true_label'])
	new.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)
	new['final_predicted_label'] = new.apply(lambda x : np.argmax([fsum(np.array([x.house_prob_x,x.house_prob_y])*w),fsum(np.array([x.land_prob_x,x.land_prob_y])*w)]) if x.predicted_label_x != x.predicted_label_y else x.predicted_label_x,axis=1)
	print('Accuracy {:.2f}%' .format((len(new[new.true_label == new.final_predicted_label]))/len(new)))
	return new
	

	
