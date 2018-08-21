from helpers import *

name = __file__.split('.')[0]
model_results = '/'.join([models, name ])

if not os.path.exists(model_results):
    os.makedirs(model_results);
    print('Created',model_results)

# Create minibatches 
params['train_mbs'] = create_mb(train_list_path, params, True)
params['valid_mbs'] = create_mb(valid_list_path, params, False)

base_model = 'VGG16_ImageNet_Caffe.model'
bm =load_model(base_model)


feature_node = find_by_name(bm, 'data')
#beforePooling_node = find_by_name(bm, 'z.x._.x._.x.x')#"z.x.x.r")
beforeDropout = find_by_name(bm,'drop7')
modelCloned = combine([beforeDropout.owner]).clone(CloneMethod.clone,{feature_node: placeholder()})


feat_norm = input_variable(params['image_dimensions'][::-1],name='input') - constant(114)
model = modelCloned(feat_norm)
drop = Dropout(.25,name='Dropout')(model)
net = Dense(params['num_classes'], init=he_normal(), activation=None,name='Dense')(drop)

model = train_and_evaluate(params,net)

plt.savefig(model_results +'/model_training_error.png',bbox_inches='tight', dpi = 200)
model.save('/'.join([model_results, name]) +'.model')
graph.plot(model, filename= model_results + "/model_graph.pdf") # Write graph visualization
	

print('Training set evaluation')
train_predictions =  eval_test_images(model,train_list_path,params['image_dimensions'])

print('Validation set evaluation')
train_predictions =  eval_test_images(model,valid_list_path,params['image_dimensions'])

predictions =  eval_test_images(model,test_list_path,params['image_dimensions'])
predictions.to_csv(model_results+'/predictions.csv')
print("Done. Wrote output to %s" % model_results+'/predictions.csv' )


graphs(predictions)
plt.savefig(model_results +'/roc_confMat.jpg', bbox_inches='tight', dpi = 200)
