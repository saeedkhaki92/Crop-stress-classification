import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression


LR1=LinearRegression(normalize=False)
LR2=LinearRegression(normalize=False)
LR3=LinearRegression(normalize=False)



X=np.load('./X_PCA_D.H.DH_3_last.npz')['data']  # Load stress metrics after applying PCA


print('concatanated',X.shape)
stress=np.load('./streesalpha3.npz')['data']

m=np.mean(X,axis=0,keepdims=True)
s=np.std(X,axis=0,keepdims=True)

X=(X-m)/s

temp=pd.read_csv('./submission_template.csv',delimiter=',')
performance_dataset=pd.read_csv('./performance_data.csv',header=0) # Contains information about: 1) Hybrids yields, environments in which hybrids were planted



all_hybrids=temp['HYBRID_ID']

Weight_matrix=np.zeros(shape=[len(all_hybrids),6]) # place in which we strore the results

def loss(Y,Yhat):

    E=Y-Yhat

    RMSE=np.sqrt(np.mean(E**2))

    return RMSE,E



for j in range(len(all_hybrids)):

    print("\rHybrid number {}/{}\n".format(j,len(all_hybrids)),end='')

    hname=all_hybrids[j]
    section=performance_dataset[performance_dataset['HYBRID_ID']==all_hybrids[j]]


    #print(section.shape)

    env_uniq=section['ENV_ID'].unique()

    print('the Hybrid was planted in %d environment' %len(env_uniq))

#print(env_id)

    ID=np.zeros(shape=[len(env_uniq)],dtype=int)
    target=np.zeros_like(ID,dtype=float)
    for i in range(len(env_uniq)):
    #print(i)
        ENV=section[section['ENV_ID']==env_uniq[i]]
    #print(ENV.shape)
        ID[i]=int(env_uniq[i][4:])
    #print(ID[i])
        target[i]=np.mean(ENV['YIELD'])##mean


    target=target.reshape(-1,1)
    #print(ID.shape)

    ID=ID-1

    XX=X[ID]  #stress metrics

    #print('DATA used',XX.shape)

    m,_=XX.shape

    LR1.fit(XX[:,0].reshape(-1,1), np.squeeze(target))

    W1 = LR1.coef_
    b1=LR1.intercept_

    LR2.fit(XX[:,1].reshape(-1,1), np.squeeze(target))

    W2 = LR2.coef_
    b2 = LR2.intercept_

    LR3.fit(XX[:,2].reshape(-1,1), np.squeeze(target))

    W3 = LR3.coef_
    b3 = LR3.intercept_


    Weight_matrix[j,:]=[W1,W2,W3,b1,b2,b3]





np.savez_compressed('./Weight_matrix_slopes_pca3_last_with_intercept',data=Weight_matrix)  # Save the slopes and intercepts of all regression lines





