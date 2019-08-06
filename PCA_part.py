import numpy as np

from sklearn.decomposition import PCA

import pandas as pd

pca1=PCA(n_components=1)


pca2=PCA(n_components=1)

pca3=PCA(n_components=1)

YS=np.load('./Syngenta/streesalpha3.npz')['data']
print(YS.shape)

X_DH=np.load('./DHcombined_7features_final.npz')['data']



mdh=np.mean(X_DH,axis=0,keepdims=True)
sdh=np.std(X_DH,axis=0,keepdims=True)

X_DH=(X_DH-mdh)/sdh

print('X_dh shape',X_DH.shape)

X_H=np.load('./heat_embedding_3features_last.npz')['data']


mh=np.mean(X_H,axis=0,keepdims=True)
sh=np.std(X_H,axis=0,keepdims=True)

X_H=(X_H-mh)/sh
print('X_Heat shape',X_H.shape)



X_D=np.load('./drought_embedding_7features_last.npz')['data']



md=np.mean(X_D,axis=0,keepdims=True)
sd=np.std(X_D,axis=0,keepdims=True)

X_D=(X_D-md)/sd
print('X_drought shape',X_D.shape)


print('DH**********')
XX_DH=pca1.fit_transform(X_DH)

print(XX_DH.shape)

print(pca1.explained_variance_ratio_)



print('D**********')

XX_D=pca2.fit_transform(X_D)

print(XX_D.shape)

print(pca2.explained_variance_ratio_)



print('H**********')

XX_H=pca3.fit_transform(X_H)

print(XX_H.shape)

print(pca3.explained_variance_ratio_)


out=np.concatenate((XX_D,XX_H,XX_DH),axis=1)
print(out.shape)

np.savez_compressed('./X_PCA_D.H.DH_3_last',data=out)  ## save the stress metrics after applying CNN




DF=pd.DataFrame(columns=['ENV_ID','Heat_metric','Drought_metric','Drought_heat_metric'])
print(np.arange(1,1561))
DF['ENV_ID']=np.arange(1,1561)


DF['Heat_metric']=np.squeeze(XX_H)
DF['Drought_metric']=np.squeeze(XX_D)
DF['Drought_heat_metric']=np.squeeze(XX_DH)

print(DF.shape)
#DF.to_csv(path_or_buf='./Stress_metrics.csv',index=False,index_label=False)