import numpy as np


import tensorflow as tf

import matplotlib.pylab as plt

from sklearn.manifold import TSNE





ENV = np.load('./soildata')['data']


print(ENV.shape)
m_e = np.mean(ENV, axis=0, keepdims=True)
s_e = np.std(ENV, axis=0, keepdims=True)

ENV = (ENV - m_e) / m_e

print(ENV.shape)

H_X = np.load('./Heat_Weather_20interval.npz')['data']  # m-by-n1   n1 is number of heat related weather variables, m is number of observations

m_h = np.mean(H_X, axis=0, keepdims=True)
s_h = np.std(H_X, axis=0, keepdims=False)

H_X = (H_X - m_h) / s_h

# H_X=np.concatenate((H_X,ENV),axis=1)   # adding the soil data to the heat variables to train heat CNN model

H_X = np.expand_dims(H_X, axis=-1)

print(H_X.shape)

D1_X = np.load('./Drought_drought_20interval.npz')['data']  # m-by-n2 #n2 is number of drought related weather variables, m is number of observations
s_d1 = np.std(D1_X, axis=0, keepdims=False)

m_d1 = np.mean(D1_X, axis=0, keepdims=True)

D1_X = (D1_X - m_d1) / s_d1
D1_X = np.concatenate((D1_X, ENV), axis=1)  # adding the soil data to the drought variables to train drought CNN model
D1_X = np.expand_dims(D1_X, axis=-1)

Y = np.load('./streesalpha3.npz')['data'].reshape(-1, 1)  # load the yield stress response, m-by-1




ID=np.argsort(np.squeeze(Y))
print(ID.shape)

I1=ID[0:len(ID)//3]
print(I1.shape)

I2=ID[len(ID)//3:2*len(ID)//3]
print(I2.shape)


I3=ID[2*len(ID)//3:]
print(I3.shape)





tf.reset_default_graph()


saver = tf.train.import_meta_graph('./Syngenta/best/stress_included_soil_drought_encoded_10_alpha3-4999.meta',clear_devices=True)  # load one of the three trained CNN models

sess=tf.Session()

saver.restore(sess,'./Syngenta/best/stress_included_soil_drought_encoded_10_alpha3-4999')

graph=tf.get_default_graph()




A=[m.values() for m in graph.get_operations()]

print(A)

is_training=graph.get_tensor_by_name('Placeholder:0')
H_t=graph.get_tensor_by_name('H_t:0')
print(H_t)
D1_t=graph.get_tensor_by_name('D_t:0')
print(D1_t)

Y_t=graph.get_tensor_by_name('Y_t:0')
print(Y_t)
out_h=graph.get_tensor_by_name('out_h_f/flatten/Reshape:0')
print("out_h",out_h)
out_d1=graph.get_tensor_by_name('out_d1_f/flatten/Reshape:0')
print("out_d1",out_d1)


#out_dh=graph.get_tensor_by_name('Relu_21:0')
#print(out_dh)


encoding=graph.get_tensor_by_name('Relu_26:0')


print('encoding',encoding)


RMSE=graph.get_tensor_by_name('RMSE:0')




x=100
I1=I1[0:x]

I2=I2[len(I2)//2-(x//2):len(I2)//2+(x//2)]


I3=I3[-x:]


X_embedded1=encoding.eval(session=sess,feed_dict={H_t:H_X[I1],D1_t:D1_X[I1],Y_t:Y[I1],is_training:False})

print("Embeded1",X_embedded1.shape)


X_embedded2=encoding.eval(session=sess,feed_dict={H_t:H_X[I2],D1_t:D1_X[I2],Y_t:Y[I2],is_training:False})

print("Embeded2",X_embedded2.shape)


X_embedded3=encoding.eval(session=sess,feed_dict={H_t:H_X[I3],D1_t:D1_X[I3],Y_t:Y[I3],is_training:False})

print("Embeded3",X_embedded3.shape)







X_embedded=encoding.eval(session=sess,feed_dict={H_t:H_X,D1_t:D1_X,Y_t:Y,is_training:False})

s=np.std(X_embedded,axis=0)
index=s>0

X_embedded=X_embedded[:,index]

print(X_embedded.shape)



np.savez_compressed('./drought_embedding_7features',data=X_embedded)  #save the stress metrics

#stand=scale()

X_embedded1=X_embedded1[:,index]
X_embedded2=X_embedded2[:,index]

X_embedded3=X_embedded3[:,index]




X_TSNE1 = TSNE(n_components=2).fit_transform(X_embedded1)

print(X_TSNE1.shape)

X_TSNE2 = TSNE(n_components=2).fit_transform(X_embedded2)

print(X_TSNE2.shape)

X_TSNE3 = TSNE(n_components=2).fit_transform(X_embedded3)

print(X_TSNE3.shape)



fig, ax = plt.subplots()

ax.scatter(X_TSNE1[:,0],X_TSNE1[:,1],c='blue',marker='^', label="Low stress",s=30)

ax.scatter(X_TSNE2[:,0],X_TSNE2[:,1],c='green',marker='o', label="Medium stress",s=30)


ax.scatter(X_TSNE3[:,0],X_TSNE3[:,1],c='red',marker='s' ,label="High stress",s=30)
ax.legend()
plt.show()


