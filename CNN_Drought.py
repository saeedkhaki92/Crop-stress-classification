import numpy as np

import tensorflow as tf

import time


def identity_block(X, f, filters, stage, block, is_training):
    epsilon = 0.001

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First component of main path

    X = tf.layers.conv1d(X, filters=F1, kernel_size=1, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name=conv_name_base + '2a', data_format='channels_last')

    param_initializers = {'beta': tf.zeros_initializer(), 'gamma': tf.ones_initializer()}

    X = tf.nn.relu(X)
    # Second component of main path (≈3 lines)

    X = tf.layers.conv1d(X, filters=F2, kernel_size=f, strides=1, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=None, name=conv_name_base + '2b', data_format='channels_last')

    X = tf.contrib.layers.batch_norm(X, decay=0.99, center=True, scale=True,
                                     updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                                     data_format='NHWC', param_initializers=param_initializers,
                                     scope=bn_name_base + '2b')

    X = tf.nn.relu(X)

    # Third component of main path (≈2 lines)

    X = tf.layers.conv1d(X, filters=F3, kernel_size=1, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=None, name=conv_name_base + '2c', data_format='channels_last')

    X = tf.nn.relu(X_shortcut + X)

    return X


def convolutional_block(X, f, filters, stage, block, is_training, s=2):
    epsilon = 0.001
    # defining name basis
    conv_name_base = 'res2' + str(stage) + block + '_branch'
    bn_name_base = 'bn2' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path

    X = tf.layers.conv1d(X, filters=F1, kernel_size=1, strides=s, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name=conv_name_base + '2a', data_format='channels_last')

    X = tf.nn.relu(X)

    X = tf.layers.conv1d(X, filters=F2, kernel_size=f, strides=1, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name=conv_name_base + '2b', data_format='channels_last')

    param_initializers = {'beta': tf.zeros_initializer(), 'gamma': tf.ones_initializer()}
    # X=tf.layers.batch_normalization(X,axis=-1,momentum=0.92,center=True,scale=True,training = is_training, name=bn_name_base+'2a')
    X = tf.contrib.layers.batch_norm(X, decay=0.99, center=True, scale=True,
                                     updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                                     data_format='NHWC', param_initializers=param_initializers,
                                     scope=bn_name_base + '2b')

    X = tf.nn.relu(X)

    X = tf.layers.conv1d(X, filters=F3, kernel_size=1, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name=conv_name_base + '2c', data_format='channels_last')

    X_shortcut = tf.layers.conv1d(X_shortcut, filters=F3, kernel_size=1, strides=s, padding='valid',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                                  name=conv_name_base + '1', data_format='channels_last')

    X = tf.nn.relu(X_shortcut + X)

    return X


def conv_heat(H_t, f0, s0, is_training):
    X = tf.layers.conv1d(H_t, filters=8, kernel_size=3, strides=s0, padding='same',
                         data_format='channels_last', activation=None,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         bias_initializer=tf.zeros_initializer())

    X = tf.nn.relu(X)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool1')
    print('Conv1 heat shape', X)

    X = identity_block(X, f0, filters=[8, 8, 8], stage=2, block='aH', is_training=is_training)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool2')

    X = convolutional_block(X, f0, filters=[8, 8, 12], stage=4, block='aH', is_training=is_training, s=2)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool52')

    X = identity_block(X, f0, filters=[10, 10, 12], stage=3, block='aH', is_training=is_training)

    # X = identity_block(X, f0, filters=[12, 12, 12], stage=4, block='aH', is_training=is_training)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool3')

    X = identity_block(X, f0, filters=[10, 10, 12], stage=5, block='aH', is_training=is_training)

    # X = convolutional_block(X, f0, filters=[12, 12, 16], stage=5, block='aH', is_training=is_training, s=2)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool5')

    return X


def conv_drought(D_t, f0, s0, is_training):
    X = tf.layers.conv1d(D_t, filters=8, kernel_size=3, strides=s0, padding='same',
                         data_format='channels_last', activation=None,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         bias_initializer=tf.zeros_initializer())

    X = tf.nn.relu(X)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool1')
    print('Conv1 drought shape', X)

    X = identity_block(X, f0, filters=[8, 8, 8], stage=2, block='aD', is_training=is_training)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool2')

    X = convolutional_block(X, f0, filters=[8, 8, 12], stage=4, block='aD', is_training=is_training, s=2)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool52')

    X = identity_block(X, f0, filters=[10, 10, 12], stage=3, block='aD', is_training=is_training)

    # X = identity_block(X, f0, filters=[12, 12, 12], stage=4, block='aH', is_training=is_training)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool3')

    # X = convolutional_block(X, f0, filters=[16, 16, 16], stage=5, block='aH', is_training=is_training, s=2)
    X = identity_block(X, f0, filters=[10, 10, 12], stage=5, block='aD', is_training=is_training)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool5')
    return X


def fully_conntected_nn(X_t, layers, is_training, name_base):
    X = tf.contrib.layers.fully_connected(inputs=X_t, num_outputs=layers[0], activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.zeros_initializer(), scope='fc1' + name_base)

    X = tf.nn.relu(X)
    # X = tf.layers.dropout(X, rate=0.3, noise_shape=[1, layers[0]], training=is_training)

    print(X)
    for i in range(1, len(layers)):
        X = tf.contrib.layers.fully_connected(inputs=X, num_outputs=layers[i], activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer(),
                                              scope='fc' + str(i + 1) + name_base)

        X = tf.nn.relu(X)
        print(X)

    return X


def main_process(H_t, D1_t, layers_dh, is_training):
    out_h = conv_heat(H_t, f0=3, s0=1, is_training=is_training)
    print('out_h', out_h)
    out_h = tf.contrib.layers.flatten(out_h, scope='out_h_f')
    print('out_h_fianl', out_h)

    out_d1 = conv_drought(D1_t, f0=3, s0=1, is_training=is_training)
    print('out_d1', out_d1)
    out_d1 = tf.contrib.layers.flatten(out_d1, scope='out_d1_f')
    print('out_d1_final', out_d1)

    # out_d2=fully_conntected_nn(D2_t, layers_d, is_training,name_base='drought')
    # print('out_d22222222222222222222', out_d2)

    # out_s = fully_conntected_nn(S_t, layers_s, is_training,name_base='soil')
    # print('out_s',out_s)

    # d=tf.concat([out_d1,out_d2],axis=1)
    # d = out_d1
    # print(d)
    ## merging d1,d2

    # DH=tf.concat([out_d1,out_h],axis=1)
    # DH=out_h
    DH = out_d1

    print("DH ********", DH)

    out_dh = fully_conntected_nn(DH, layers_dh, is_training, name_base='DH')

    print('out_dh*********', out_dh)

    return out_dh


def kfold(data,indice,k):

    #I=np.random.permutation(data.shape[0])
    if k == 1:

        folds=[data]
    else:
        data=data[indice]

        length = int(data.shape[0] / k)  # length of each fold
        folds = []

        for i in range(k - 1):
            folds += [data[i * length:(i + 1) * length]]
        folds += [data[(k - 1) * length:]]

    return folds

def cost_function(Yhat, Y):
    E = (Yhat - Y) ** 2
    MSE = tf.squeeze(tf.reduce_mean(E))

    RMSE = tf.pow(MSE, 0.5)
    Loss = tf.losses.huber_loss(Y, Yhat, weights=1.0, delta=1.0)

    return RMSE, Loss


def main_function(H_X, D1_X, Y, layers_dh, batch_size_tr, batch_size_te, learning_rate, max_it, p):
    t1=time.time()

    m=H_X.shape[0]
    k=10
    #v=int(p*m)
    #np.random.seed(100)
    indices=np.random.permutation(m)

    #H_X_test=H_X[indices[0:v]]
    #H_X_training=H_X[indices[v:]]
    folds_heat = kfold(H_X, indices, k)
    #print(H_X_test.shape)
    #print(H_X_training.shape)
    #D1_X_test=D1_X[indices[0:v]]
    #D1_X_training=D1_X[indices[v:]]
    folds_drought1 = kfold(D1_X, indices, k)
    #print(D1_X_test.shape)
    #print(D1_X_training.shape)

    #D2_X_test = D2_X[indices[0:v]]
    #D2_X_training = D2_X[indices[v:]]
    folds_drought2 = kfold(D2_X, indices, k)
    #print(D2_X_test.shape)
    #print(D2_X_training.shape)

    #S_X_test = S_X[indices[0:v]]
    #S_X_training = S_X[indices[v:]]
    folds_soil = kfold(S_X, indices, k)
    #print(S_X_test.shape)
    #print(S_X_training.shape)

    #Y_training=Y[indices[v:]]
    #Y_test=Y[indices[0:v]]
    folds_Y=kfold(Y,indices,k)
    #print(Y_test.shape)
    #print(Y_training.shape)


    print(len(folds_Y))

    rmse_tr_all=[]
    rmse_te_all=[]
    for f in range(k):
        tf.reset_default_graph()

        print('**************fold %d ******************' %(f+1))

        new_folds_heat = folds_heat.copy()

        H_X_test = new_folds_heat[f]

        del new_folds_heat[f]

        H_X_training = np.vstack(new_folds_heat)


        new_folds_drought1 = folds_drought1.copy()

        D1_X_test = new_folds_drought1[f]

        del new_folds_drought1[f]

        D1_X_training = np.vstack(new_folds_drought1)


        new_folds_drought2 = folds_drought2.copy()

        D2_X_test = new_folds_drought2[f]

        del new_folds_drought2[f]

        D2_X_training = np.vstack(new_folds_drought2)


        new_folds_soil = folds_soil.copy()

        S_X_test = new_folds_soil[f]

        del new_folds_soil[f]

        S_X_training = np.vstack(new_folds_soil)


        new_folds_Y = folds_Y.copy()

        Y_test = new_folds_Y[f]

        del new_folds_Y[f]

        Y_training = np.vstack(new_folds_Y)

        print(Y_training.shape)
        print(Y_test.shape)
        m1=np.mean(np.squeeze(Y_training))

        rte=np.sqrt(np.mean((np.squeeze(Y_test)-m1)**2))
        rtr=np.sqrt(np.mean((np.squeeze(Y_training)-m1)**2))
        print('STD of Y train and Y test are  %f %f' % (rtr,rte))



        with tf.device('/cpu:0'):

            H_t = tf.placeholder(dtype=tf.float32, shape=(None, 60+0, 1), name='H_t')

            D1_t = tf.placeholder(dtype=tf.float32, shape=(None, 70+16, 1), name='D_t')

            D2_t = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='D2_t')

            S_t = tf.placeholder(dtype=tf.float32, shape=[None, 14], name='S_t')

            Y_t=tf.placeholder(dtype=tf.float32,shape=[None,1],name='Y_t')

            is_training=tf.placeholder(dtype=tf.bool)

            Yhat = main_process(H_t, D1_t, D2_t, S_t, layers_d, layers_s, layers_dh, layers_dh_s, is_training)


            Yhat=tf.identity(Yhat,name='Yhat')


            with tf.name_scope('Loss'):

                RMSE,Loss=cost_function(Yhat, Y_t)


            RMSE=tf.identity(RMSE,name='RMSE')


            with tf.name_scope('train'):

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Loss)

                sess = tf.Session()

                sess.run(tf.global_variables_initializer())

            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                print(variable)
                shape = variable.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    #   print(dim)
                    variable_parameters *= dim.value
                # print(variable_parameters)
                total_parameters += variable_parameters
            print("total_parameters", total_parameters)

            for i in range(max_it):



                I=np.random.choice(S_X_training.shape[0],size=batch_size_tr)

                batch_H=H_X_training[I]
                batch_D1=D1_X_training[I]
                batch_D2=D2_X_training[I]
                batch_S=S_X_training[I]
                batch_Y=Y_training[I]


                sess.run(train_op,feed_dict={H_t:batch_H,D1_t:batch_D1,D2_t:batch_D2,S_t:batch_S,Y_t:batch_Y,is_training:True})


                if i%400==0:


                    #rmse_tr=sess.run(RMSE, feed_dict={H_t: batch_H, D1_t: batch_D1, D2_t: batch_D2, S_t: batch_S, Y_t: batch_Y,
                     #                             is_training: False})

                    I1 = np.random.randint(S_X_test.shape[0], size=batch_size_te)

                    batch_H_te = H_X_test[I1]
                    batch_D1_te = D1_X_test[I1]
                    batch_D2_te = D2_X_test[I1]
                    batch_S_te = S_X_test[I1]
                    batch_Y_te = Y_test[I1]

                    #rmse_te = sess.run(RMSE,
                     #                  feed_dict={H_t: batch_H_te, D1_t: batch_D1_te, D2_t: batch_D2_te, S_t: batch_S_te, Y_t: batch_Y_te,
                      #                            is_training: False})

                    rmse_tr = sess.run(RMSE,
                                       feed_dict={H_t: H_X_training, D1_t: D1_X_training, D2_t: D2_X_training,
                                                  S_t: S_X_training,
                                                  Y_t: Y_training,
                                                  is_training: False})

                    rmse_te = sess.run(RMSE,
                                       feed_dict={H_t: H_X_test, D1_t: D1_X_test, D2_t: D2_X_test, S_t: S_X_test,
                                                  Y_t: Y_test,
                                                  is_training: False})

                    print("Iteration %d , The training RMSE is %f  and test RMSE is %f " % (i, rmse_tr, rmse_te))
                    print(H_X_test.shape)
                    print(D1_X_training.shape)
            rmse_tr = sess.run(RMSE,
                               feed_dict={H_t: H_X_training, D1_t: D1_X_training, D2_t: D2_X_training, S_t: S_X_training,
                                          Y_t: Y_training,
                                          is_training: False})

            rmse_te = sess.run(RMSE,
                               feed_dict={H_t: H_X_test, D1_t: D1_X_test, D2_t: D2_X_test, S_t: S_X_test,
                                          Y_t: Y_test,
                                          is_training: False})

        saver = tf.train.Saver()
        saver.save(sess, './Syngenta/kfold/stress_included_soil_drought_entire_encoded_10_alpha3_last_fold'+str(f), global_step=i)  # Saving the model

        rmse_te_all.append(rmse_te)
        rmse_tr_all.append(rmse_tr)

    t2=time.time()


    print("training time", t2 - t1)
    return rmse_te_all,rmse_tr_all


ENV = np.load('./soildata')['data']


print(ENV.shape)


print(ENV.shape)

H_X = np.load('./Heat_Weather_20interval.npz')['data']  # m-by-n1   n1 is number of heat related weather variables, m is number of observations




# H_X=np.concatenate((H_X,ENV),axis=1)   # adding the soil data to the heat variables to train heat CNN model

H_X = np.expand_dims(H_X, axis=-1)

print(H_X.shape)

D1_X = np.load('./Drought_drought_20interval.npz')['data']  # m-by-n2 #n2 is number of drought related weather variables, m is number of observations

D1_X = np.concatenate((D1_X, ENV), axis=1)  # adding the soil data to the drought variables to train drought CNN model
D1_X = np.expand_dims(D1_X, axis=-1)

Y = np.load('./streesalpha3.npz')['data'].reshape(-1, 1)  # load the yield stress response, m-by-1

batch_size_tr = 50
batch_size_te = 78
learning_rate = 0.0002
max_it = 5000
p = 0.05

layers_dh = [10, 1]

is_training = True

rmse_te_all,rmse_tr_all = main_function(H_X, D1_X, Y, layers_dh, batch_size_tr, batch_size_te, learning_rate, max_it, p)
print('train')
print(rmse_tr_all)
print('test')
print(rmse_te_all)



print('train RMSE is %f and test RMSE is %f ' %(np.mean(rmse_tr_all),np.mean(rmse_te_all)))

































