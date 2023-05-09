import pandas as pd
import os
import torch
import pickle
import torch
import pandas as pd

#df = pd.DataFrame(columns=['seed','dataset','model','loss','train_accuracy','validation_accuracy','test_accuracy'])
df=pd.read_pickle('results/performances.pkl')




#data=pd.read_csv('results/Citeseer/GCN.csv')
#print(data.columns)
#data=data.drop('Unnamed: 0.1',axis=1)
#data.to_csv('results/Citeseer/GCN.csv')
#data=pd.read_csv('results/Citeseer/LTFGW.csv')
#print(data.columns)
#data.insert(4,'validation_accuracy',[0,0,0])
#print(data.columns)
#data.to_csv('results/Citeseer/LTFGW.csv')
#new_data=data.drop('Unnamed: 0',axis=1)
#new_data.to_csv('results/Citeseer/GCN.csv')

#new_data=data.drop('Unnamed: 0',axis=1)
#new_data2=new_data.drop('Unnamed: 0.1',axis=1)
#new_data2.to_csv('results/Citeseer/GCN.csv')


#df1=pd.read_pickle('results/LTFGW/Citeseer.pkl')
#print(df1['max_val_accuracy'])


#df=pd.read_pickle('results/LTFGW/Toy_graph.pkl')
#print(df)

#df=pd.DataFrame(columns=['seed','loss','train_accuracy','validation_accuracy','test_accuracy','max_val_accuracy'])
#df.to_pickle('results/GCN/Citeseer.pkl')

#df=pd.read_pickle('results/GCN/Citeseer.pkl')
#row={'seed':1, 'loss': [0],'train_accuracy':[0] ,'validation_accuracy': [0],'test_accuracy':0}
#df.loc[len(df)]=row
#df.to_pickle('results/LTFGW/Citeseer.pkl')
#dataset=Citeseer_data()

#print(dataset.val_mask)


#torch.save(model.state_dict(),filename_model)

