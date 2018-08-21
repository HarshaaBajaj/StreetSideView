from helpers import *

name = __file__.split('.')[0]
model_results = '/'.join([models, name ])

if not os.path.exists(model_results):
    os.makedirs(model_results);
    print('Created',model_results)

params['image_dimensions'] = (227,227,3)
model= load_model('/'.join([model_results, name]) +'.model')

print('Training set evaluation')
train_predictions =  eval_test_images(model,train_list_path,params['image_dimensions'])

print('Validation set evaluation')
train_predictions =  eval_test_images(model,valid_list_path,params['image_dimensions'])

predictions =  eval_test_images(model,test_list_path,params['image_dimensions'])
predictions.to_csv(model_results+'/predictions.csv')
print("Done. Wrote output to %s" % model_results+'/predictions.csv' )


graphs(predictions)
plt.savefig(model_results +'/roc_confMat.jpg', bbox_inches='tight', dpi = 200)
