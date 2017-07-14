# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:30:25 2017

@author: hnozadi
"""
'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import (EarlyStopping, 
                             LearningRateScheduler, 
                             ModelCheckpoint,
                             History)
from keras.regularizers import l2, activity_l2

import time
import glob
from sklearn.cross_validation import train_test_split, LeaveOneOut
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.decomposition import RandomizedPCA, KernelPCA, PCA, FastICA
from sklearn.svm import SVC
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import PredefinedSplit
import scipy.io

def meshFeatures(pointnb,nb_s,listdir):
    # read SPHARM result for label i
    # read a vtk file and save x y z in each row of the matrix data.
    data_feature = np.zeros((nb_s , pointnb*3)) # list x,y,z one after another
    subjCtr = 0             
    for subj in range(0, nb_s):
        
        input_file = listdir[subj]
        print (input_file)
        with open(input_file, 'r')  as f:  
            l=f.readline()
            l=f.readline()
            l=f.readline()
            l=f.readline()
            l=f.readline()
            print (l)
            ind1=l.find('POINTS ')
            ind2=l.find('float')
            ind_start = ind1 +7;
            ind_end = ind2;
            pointnb = int (l[ind_start:ind_end])
            print(pointnb)
            linenb = int(pointnb/3)
            linenb_extra = pointnb%3
            data = np.zeros((pointnb,3))
            
            ctrln = 0
            ctrlnf = 0
            for ln in range(0,linenb):
                l=f.readline()
                temp = np.fromstring(l, dtype=float, sep='\t')
                data[ctrln,:] = temp [0:3]
                ctrln = ctrln + 1
                data[ctrln,:] = temp [3:6]
                ctrln = ctrln + 1
                data[ctrln,:] = temp [6:9]
                ctrln = ctrln + 1
                for j in range (0, 9):
                    data_feature[subjCtr,ctrlnf] = temp[j]
                    ctrlnf = ctrlnf +1
            if (linenb_extra==1):
                l=f.readline()
                temp = np.fromstring(l, dtype=float, sep='\t')
                data[ctrln,:] = temp [0:3]
                ctrln = ctrln + 1
                for j in range (0, 3):
                    data_feature[subjCtr,ctrlnf] = temp[j]
                    ctrlnf = ctrlnf +1
            if (linenb_extra==2):
                l=f.readline()
                temp = np.fromstring(l, dtype=float, sep='\t')
                data[ctrln,:] = temp [0:3]
                ctrln = ctrln + 1
                data[ctrln,:] = temp [3:6]
                ctrln = ctrln + 1
                for j in range (0, 6):
                    data_feature[subjCtr,ctrlnf] = temp[j]
                    ctrlnf = ctrlnf +1
                
            print(data_feature)# includes one value in each row
            print (data) # includes x,y,z in each row
            subjCtr = subjCtr +1
    print(data_feature)
    return data_feature
if __name__ == '__main__':
    n_classes = 2
    # ADNI dataset
    pathADNL='/Users/mashad/database/AD/shape_outputData_all/spectralM_NC_AD_17/vtk/';
    pathEMCINL='/Users/mashad/database/AD/shape_outputData_all/spectralM_EMCI_17/vtk/';
    pathLMCINL='/Users/mashad/database/AD/shape_outputData_all/spectralM_LMCI_17/vtk/';
    pointnb = 2170
    # 50/50/50 dataset
#    pathADNL='/Users/mashad/database/ADNI/comparison/shapeAnalysis_Diffeo/reconstructed_meshes/NL_AD_17/';
#    pathMCINL='/Users/mashad/database/ADNI/comparison/shapeAnalysis_Diffeo/reconstructed_meshes/NL_MCI_17/';
#    pointnb = 2640
    #-------------- Mesh coordicates ------------------------------------------
    # Read NC 17 from AD_NC_17
    listdir = glob.glob(pathADNL+'groupA_*_pp_surf_rec.vtk')
    print  (listdir)
    nb_s_NC = len (listdir)
    data_NC = meshFeatures(pointnb,nb_s_NC, listdir)
    # Read AD 17 from AD_NC_17
    listdir = glob.glob(pathADNL+'groupB_*_pp_surf_rec.vtk')
    print  (listdir)
    nb_s_AD = len (listdir)
    data_AD = meshFeatures(pointnb,nb_s_AD, listdir)
    # Read MCI 17 from MCI_NC_17
    listdir = glob.glob(pathLMCINL+'groupB_*_pp_surf_rec.vtk')
    print  (listdir)
    nb_s_MCI = len (listdir)
    data_MCI = meshFeatures(pointnb,nb_s_MCI, listdir)
    # -----------Read EMCI 17 for EMCI/LMCI comparison-----------------------------------
#    listdir = glob.glob(pathEMCINL+'groupB_*_pp_surf_rec.vtk')
#    print  (listdir)
#    nb_s_EMCI = len (listdir)
#    data_EMCI = meshFeatures(pointnb,nb_s_EMCI, listdir)
    #-------------- curvature -------------------------------------------------
    # extract curvature from diffeoSpectralmatching/code_curvature
#    data_NC = np.genfromtxt(pathADNL+'Cg_NL_17.csv',delimiter=',')
#    data_AD = np.genfromtxt(pathADNL+'Cg_AD_17.csv',delimiter=',')
#    data_MCI = np.genfromtxt(pathMCINL+'Cg_MCI_17.csv',delimiter=',')
    #--------------------------------------------------------------------------
    nb_s_NC = data_NC.shape[0]
    nb_s_AD = data_AD.shape[0]
    nb_s_MCI = data_MCI.shape[0]
#    nb_s_EMCI = data_EMCI.shape[0]
    # MCI NC:---------
    data = np.vstack((data_NC,data_MCI))
    # AD NC:----------
#    data = np.vstack((data_NC,data_AD))
    # AD MCI:----------
#    data = np.vstack((data_AD,data_MCI))
    # EMCI LMCI :-----------
#    data = np.vstack((data_EMCI,data_MCI))
    # ALL :-----------
#    tmp = np.vstack((data_NC,data_AD))
#    data = np.vstack((tmp,data_MCI))
#    nb_s = data.shape[0]
    #create target:
    
    target_NC = 1*np.ones((nb_s_NC,1))
    target_AD = 2*np.ones((nb_s_AD,1))
    target_MCI = 3*np.ones((nb_s_MCI,1))
    
    # MCI NC:----------
    target_MCI = 2*np.ones((nb_s_MCI,1))
    target = np.vstack((target_NC,target_MCI))
    # AD NC:----------
#    target = np.vstack((target_NC,target_AD))
    # AD MCI : -------
#    target_MCI = 1*np.ones((nb_s_MCI,1))
#    target = np.vstack((target_AD,target_MCI))
    #EMCI LMCI : -------
#    target_EMCI = 1*np.ones((nb_s_EMCI,1))
#    target_MCI = 2*np.ones((nb_s_MCI,1))
#    target = np.vstack((target_EMCI,target_MCI))
    #All:-------------
#    tmpTarget = np.vstack((target_NC,target_AD))
#    target = np.vstack((tmpTarget,target_MCI))
    
    target = target.astype(int)
    np.savetxt("/Users/mashad/database/ADNI/classify/target.csv", target, delimiter=",")
    np.savetxt("/Users/mashad/database/ADNI/classify/data.csv", data, delimiter=",")
    
    print("------------permutation starts-------------")
    X,  y = shuffle(data, target, random_state=0)# shuffles the rows. random_state == seed

#    target_names = numpy.array(['BECTS','Controls'], dtype=object)
#    n_classes = target_names.shape[0]
    #########Scale - Normalization #################################################
    print("------------Normalization starts-----------")
    #scale:
    scaler = preprocessing.StandardScaler().fit(X) #way1'  
    X_scaled = scaler.transform(X) 

    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42) # default 42
    
    #########---------------------- VAE --------------------------------------------
    batch_size = 28
    original_dim = data.shape[1] #784
    latent_dim = 128 # 128
    intermediate_dim1 = 512 # 512
#    intermediate_dim2 = 512
    epsilon_std = 0.01
    nb_epoch = 2000 # default = 40
    kl_coef = 0
#    trainSize = 107  # Total 112
#    x_train_set = x_train[:trainSize,:]
#    y_train_set = y_train[:trainSize,:]
#    x_val_set = x_train[trainSize:,:]
#    y_val_set = y_train[trainSize:,:]

    x = Input(shape=(original_dim,), name='input')
    h1 = Dense(intermediate_dim1, activation='relu', init='lecun_uniform', W_regularizer=l2(0.05))(x)
    h1_do = Dropout(0.5)(h1)
#    h = Dense(intermediate_dim2, activation='relu', init='lecun_uniform', W_regularizer=l2(0.05))(h1_do)
#    h_do = Dropout(0.5)(h)
    z_mean = Dense(latent_dim)(h1_do)
    z_log_std = Dense(latent_dim)(h1_do)
    
    def sampling(args):  # Mahsa :  adding noise
        z_mean, z_log_std = args
        epsilon = K.random_normal(shape=z_mean.shape,
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_std) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_std])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
    z_do = Dropout(0.5)(z)
    # we instantiate these layers separately so as to reuse them later
#    decoder_h2 = Dense(intermediate_dim2, activation='relu', init='lecun_uniform', W_regularizer=l2(0.05))
#    h2_decoded = decoder_h2(z_do)
#    h2_decoded_do = Dropout(0.5)(h2_decoded)
    
    decoder_h = Dense(intermediate_dim1, activation='relu', init='lecun_uniform', W_regularizer=l2(0.05))
    h_decoded = decoder_h(z_do)
    
    decoder_mean = Dense(original_dim, activation='relu', name='vae_output', init='lecun_uniform', W_regularizer=l2(0.05))
    x_decoded_mean = decoder_mean(h_decoded)
    #------------MLP layers --------------------------------------------------------
    def combine_mean_std(args):
        z_mean, z_log_std = args
        return z_mean + K.exp(z_log_std)
    
    MLP_in = Lambda(combine_mean_std, output_shape=(latent_dim,))([z_mean, z_log_std])
    MLP_in_do = Dropout(0.5)(MLP_in)
    MLP1 = Dense (latent_dim, activation='relu', init='lecun_uniform', W_regularizer=l2(0.05))(MLP_in_do)#, W_regularizer=l2(0.01)
    sftmx = Dense(n_classes, activation='softmax', name='classification_out')(MLP1)
#    #--------------------------------------------------------------------------------
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean) #
        kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
        return K.mean(xent_loss + kl_loss)


    # Other callbacks
    vae = Model(input=x, output=[x_decoded_mean, sftmx])
    vae.compile(optimizer='rmsprop', loss={'vae_output': vae_loss, 'classification_out': 'categorical_crossentropy'}, metrics={'classification_out': 'accuracy'})
    
    #change learning rate from 0.0001 to 0.00001 in a logarithmic way------------------------ 
    LearningRate = np.logspace(-5, -6, num=nb_epoch)
    LearningRate = LearningRate.astype('float32')
    vae.optimizer.lr.set_value(LearningRate[0])    
    def scheduler(epoch):
        
        vae.optimizer.lr.set_value(LearningRate[epoch-1])
        return float(vae.optimizer.lr.get_value())

    change_lr = LearningRateScheduler(scheduler)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath="/Users/mashad/python_projects/weights_resunet_"+ timestr + ".hdf5"
    early_stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto') #Stop training when a monitored quantity has stopped improving.
    checkpointer = ModelCheckpoint(filepath="/Users/mashad/python_projects/weights_resunet_"+ timestr + ".hdf5", verbose=1, save_best_only=True)#Save the model after every epoch.    
    print ("-------------vae.fit starts----------------")
    from keras.utils.np_utils import to_categorical
    history = History()
    fitlog = vae.fit({'input': x_train}, {'vae_output': x_train, 'classification_out': to_categorical(y_train-1)},
            shuffle=True,
            nb_epoch=100,
            batch_size=batch_size,
            validation_split=0.2, validation_data=None, callbacks=[history]) #callbacks=[early_stopping, checkpointer,change_lr]
    print (history.history)
    plt.title('loss')
    plt.plot(fitlog.epoch, fitlog.history['loss'], 'r-')
    plt.plot(fitlog.epoch, fitlog.history['val_loss'], 'b-')
    plt.show()
    plt.title('classification_out_loss')
    plt.plot(fitlog.epoch, fitlog.history['classification_out_loss'], 'r-')
    plt.plot(fitlog.epoch, fitlog.history['val_classification_out_loss'], 'b-')
    plt.show()
    plt.title('vae_output_loss')
    plt.plot(fitlog.epoch, fitlog.history['vae_output_loss'], 'r-')
    plt.plot(fitlog.epoch, fitlog.history['val_vae_output_loss'], 'b-')
    plt.show()
    plt.title('classification_out_acc')
    plt.plot(fitlog.epoch, fitlog.history['classification_out_acc'], 'r-')
    plt.plot(fitlog.epoch, fitlog.history['val_classification_out_acc'], 'b-')
    plt.show()

    

    
    # -----------evaluate --------------------------
    
#    vae.load_weights(filepath)
    score = vae.evaluate({'input': x_test}, {'vae_output': x_test, 'classification_out': to_categorical(y_test-1)}, batch_size=batch_size)
    print ("-----------Best metrics: ")
    print(vae.metrics_names)
    print (" ", score)
    
    # build a model to project inputs on the latent space  # Mahsa: this is dimensionality reduction
    print ("encoder model starts")    
    encoder = Model(x, z_mean)
    # ----------display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()
    
    # display 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], c=y_test)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
#    # build a digit generator that can sample from the learned distribution
#    decoder_input = Input(shape=(latent_dim,))
#    decoder_input2 = decoder_h2(decoder_input)
#    _h_decoded = decoder_h(decoder_input2)
#    _x_decoded_mean = decoder_mean(_h_decoded)
#    generator = Model(decoder_input, _x_decoded_mean)
    
   
    
#    # display a 2D manifold of the digits
#    n = 15  # figure with 15x15 digits
#    digit_size = 28
#    figure = np.zeros((digit_size * n, digit_size * n))
#    # we will sample n points within [-15, 15] standard deviations
#    grid_x = np.linspace(-15, 15, n)
#    grid_y = np.linspace(-15, 15, n)
#    
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):
#            z_sample = np.array([[xi, yi]]) * epsilon_std
#            x_decoded = generator.predict(z_sample)
#            digit = x_decoded[0].reshape(digit_size, digit_size)
#            figure[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit
#    
#    plt.figure(figsize=(10, 10))
#    plt.imshow(figure)
#    plt.show()
#    #----------------------- SVM ----------------------------------------------
#    #prepare y_train:
#    y_train2=np.zeros((y_train.shape[0]))
#    for row in range(0,y_train.shape[0]):
#        y_train2[row] = y_train[row,0]
#    print(y_train2 )
#        # SVM-RBF:
#    param_grid = {'C': [1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [1e-09,1e-08,1e-07,1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1e1, 1e2, 1e3],  }
##    skf = StratifiedKFold(y_train, n_folds=3)
#                    
#    clf = GridSearchCV(SVC(kernel='rbf',decision_function_shape='ovr'), param_grid=param_grid)  # remove , class_weight='balanced'
##        # SVM-Linear
##            param_grid = {'C': [1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5] }
##            clf = GridSearchCV(SVC(kernel='linear'), param_grid)  # remove , class_weight='balanced'
#    x_train_encoded = encoder.predict(x_train, batch_size=batch_size)  ### correct????????
#    clf = clf.fit(x_train_encoded, y_train2)   
#    print("Best parameters set found on development set:")
#    print(clf.best_params_)
#    print()
#    y_pred = clf.predict(x_test_encoded)
#    print("y_pred:\n", y_pred)                
#    # prepare y_test
#    y_test2=np.zeros((y_test.shape[0]))
#    for row in range(0,y_test.shape[0]):
#        y_test2[row] = y_test[row,0]
#    
#    print("y_test:\n", y_test2)
variational_autoencoder_8_2.py
Open with
Displaying variational_autoencoder_8_2.py.