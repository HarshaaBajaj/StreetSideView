from functools import reduce
import matplotlib.pyplot as plt
from itertools import *
from math import *
from sklearn.metrics import *

def ensemble(dfs,w=[.33,.33,.33]):
    #a,v,r
    assert(ceil(fsum(w))==1)
    df_final = reduce(lambda left,right: pd.merge(left,right,on=['image_path','true_label']), dfs)
    df_final.insert(2,'final_predicted_label',0)
    df_final['final_predicted_label'] = df_final.apply(lambda x : np.argmax([fsum(np.array([x.house_prob_x,x.house_prob_y,x.house_prob])*w),fsum(np.array([x.land_prob_x,x.land_prob_y,x.land_prob])*w)]),axis=1)
    print('Accuracy {:.2f}%' .format(sum(df_final.true_label == df_final.final_predicted_label)/len(df_final)))
    return df_final

p = '../../models/'
res =['AlexNet/', 'VGG_16/' , 'ResNet_18/']
dfs = list(map(lambda x : pd.read_csv(p + x +'predictions.csv',index_col=0), res))

ac = list(map(lambda x :print('Accuracy {:.2f}%' .format(sum(x.true_label == x.predicted_label)/len(x))) , dfs))

w = np.array([.2,.1,.7])

new = ensemble(dfs,w)
new.to_csv(p + 'final.csv')
graphs(temp)