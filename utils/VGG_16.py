from helpers import *

name = __file__.split('.')[0]
model_results = '/'.join([models, name ])

if not os.path.exists(model_results): os.makedirs(model_results);
    
try : model= load_model('/'.join([model_results, name]) +'.model')
 
except:
	# Create minibatches 
	params['train_mbs'] = create_mb(train_list_path, params, True)
	params['valid_mbs'] = create_mb(valid_list_path, params, False)
	
	params['epoch_size_train'] = int(fsum(1 for line in open(train_list_path)))
	params['epoch_size_test'] = int(fsum(1 for line in open(valid_list_path)))
    

	model = train_and_evaluate(params, model=create_vgg)
	plt.savefig(model_results +'/model_training_error.png',bbox_inches='tight', dpi = 200)
	model.save('/'.join([model_results, name]) +'.model')
	graph.plot(model, filename= model_results + "/model_graph.pdf") # Write graph visualization

# to make sure training set is predicted well
#print('Training set evaluation')
#train_predictions =  eval_test_images(model,train_list_path,params['image_dimensions'])

#print('Validation set evaluation')
#train_predictions =  eval_test_images(model,valid_list_path,params['image_dimensions'])

predictions =  eval_test_images(model,test_list_path,params['image_dimensions'])
predictions.to_csv(model_results+'/predictions.csv')
#print("Done. Wrote output to %s" % model_results+'/predictions.csv' )


graphs(predictions)
plt.savefig(model_results +'/roc_confMat.jpg', bbox_inches='tight', dpi = 200)
