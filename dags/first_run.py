import datetime as dt

from airflow import DAG
from airflow.operators import PythonOperator
from airflow.models import Variable

import sys,os
sys.path.append(os.path.abspath('..'))
from utils.helpers import *


input_file_path = Variable.get('input_file')
training = Variable.get('is_training')

default_args = {
    'owner': 'Newline Financial',
    'depends_on_past': False,
    'start_date': dt.datetime.now(),
    'provide_context': True
}

dag = DAG('property_classification', default_args=default_args)

fetch_images_task = PythonOperator(task_id = 'fetch_images', python_callable = fetch_images, dag=dag,
								   op_kwargs = {'input_file':input_file_path,'is_training':training})

alex_net_task = PythonOperator(task_id = 'alex_net', python_callable = alex, dag=dag)								   
vgg_task = PythonOperator(task_id = 'vgg', python_callable = vgg, dag=dag)
res_net_task = PythonOperator(task_id = 'res_net', python_callable = resnet, dag=dag)

ensemble_task = PythonOperator(task_id = 'ensemble', python_callable = final_predictions, dag=dag)

# define the relationship between the tasks 
fetch_images_task.set_downstream(alex_net_task)
alex_net_task.set_downstream(vgg_task)
vgg_task.set_downstream(res_net_task)
res_net_task.set_downstream(ensemble_task)

#todo: check run time for parallel execution of a,v,r