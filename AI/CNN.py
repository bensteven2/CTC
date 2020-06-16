from __future__ import absolute_import, division, print_function, unicode_literals
#from skimage import io, transform
import glob
import os
import sys
import numpy as np
import time
import xlrd
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter
import cv2
# import xlwt
import tensorflow as tf

from keras import losses
# from tensorflow.keras import layers
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_predict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
###
#import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras import regularizers
###
#sys.path.append("..")
#print('the path of common is {}'.format(sys.path))
import common.draw_roc as draw_roc 
from AI.build_model import * 



def train_by_all(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2,roc_address="roc.pdf",roc_title="ROC curve"):
    i = 0
    print("-------------------------in the beginning of train_by_all_trainning_set:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    if True:
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images), np.array(train_images)
        Y_train, Y_test = np.array(train_labels), np.array(train_labels)

        Y_train = to_categorical(Y_train, num_classes=None)
        Y_test = Y_train
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("label_types:",label_types)
        model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
        print("|||||||||||||||before compile:")
        # 编译模型
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        #from keras import losses
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ### optimizer='sgd' optimizer='adam'
        #mean_squared_error mean_absolute_percentage_error mean_absolute_error mean_squared_logarithmic_error
        #squared_hinge hinge categorical_hinge logcosh
        # categorical_crossentropy+adam == very good
        #model.compile(loss=losses.categorical_crossentropy, optimizer='adam')
        #model.compile(loss=losses.mean_squared_error, optimizer='adam')
        model.compile(loss=losses.mean_squared_error, optimizer='sgd',metrics=['accuracy'])
        ##below 
        #model.compile(loss=losses.mean_squared_error,  optimizer='adam',metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        #now change to common draw_roc_curve(Y_test[:,1], probas_[:, 1],roc_address,roc_title=roc_title)
        auc_title = ""
        draw_roc.draw_roc_curve(Y_test[:,1], probas_[:, 1],roc_address,auc_title,roc_title=roc_title)
        print("roc_address: ",roc_address)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    del model  # deletes the existing model
    print("-------------after del model")

def cross_validation_1(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2, roc_address="roc_png"):
    
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in cross_validation_1:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    print("label_types:",label_types)
    model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]

        Y_train = to_categorical(Y_train, num_classes=None)
        Y_test = to_categorical(Y_test, num_classes=None)
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        # 编译模型
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        from keras import losses
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ### optimizer='sgd' optimizer='adam'
        #mean_squared_error mean_absolute_percentage_error mean_absolute_error mean_squared_logarithmic_error
        #squared_hinge hinge categorical_hinge logcosh
        # categorical_crossentropy+adam == very good
        model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
        #model.compile(loss=losses.mean_squared_error, optimizer='adam')
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ##below 
        #model.compile(loss=losses.mean_squared_error,  optimizer='adam',metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test[:,1], probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        #print("Y_test:",Y_test)
        #print("probas_[:, 0]:",probas_[:, 0])
        #print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(roc_address)
    plt.show()

    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    #del model  # deletes the existing model
    print("-------------after del model")
    return mean_auc,model

def cross_validation_by_model_3(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title):
    
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in cross_validation_by_model_3:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    print("label_types:",label_types)
    #model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
    #model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
    model=create_model(model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1, L2=L2, F1=F1, F2=F2, F3=F3)
    #model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1, L2=L2, F1=F1, F2=F2, F3=F3, epochs=epochs, batch_size=2, verbose=1)
    pdf = PdfPages(roc_address)         #先创建一个pdf文件
    plt.figure
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]

        #Y_train = to_categorical(Y_train, num_classes=None)
        #Y_test = to_categorical(Y_test, num_classes=None)
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test[:,1], probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        #print("Y_test:",Y_test)
        #print("probas_[:, 0]:",probas_[:, 0])
        #print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)
    plt.legend(loc="lower right")
    plt.savefig(roc_address)
    plt.show()
    pdf.savefig()                            #将图片保存在pdf文件中
    plt.close()
    pdf.close()
    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    #del model  # deletes the existing model
    print("-------------after del model")
    return mean_auc,model

def Cross_validation(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2, roc_address="roc.png"):
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in Cross_validation:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("label_types:",label_types)
        model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
        print("|||||||||||||||before compile:")
        # 编译模型
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        #print("-----------------------tf.config.device_count333333:",tf.config.device_count)
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        print("Y_test:",Y_test)
        print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_model_address + "ROC_validation.png")
    plt.show()

    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    del model  # deletes the existing model
    print("-------------after del model")

    
def Keras_Classifier(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3):
    print("begin of keras_classifier: ")
    model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times,L1=L1,L2=L2,F1=F1,F2=F2,F3=F3, epochs=epochs, batch_size=2, verbose=1)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=5000)
    #scores = cross_val_score(model1, train_images, train_labels, cv=kfold)
    print("before of cross_val_predict:")
    y_pre = cross_val_predict(model, train_images, train_labels, cv=kfold)
    print(y_pre)
    y_scores = y_pre
    print("train_labels",train_labels)
    print("y_scores",y_scores)
    fpr, tpr, thresholds = roc_curve(train_labels, y_scores)
    plt.plot(fpr, tpr)
    plt.savefig("ROC.png")
    plt.show()


def cross_validation_select_parameters_3(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title):
    print("begin of crosss_validation_select_parameters_3: ")
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    # encode class values as integers
    #encoder = LabelEncoder()
    #Y_train_encoder = encoder.fit_transform(Y_train)
    # convert integers to  variables (one hot encoding)
    Y_train = to_categorical(Y_train, 2)
    print("Y_train:", Y_train)    
    for times in range(0,10):
        L1_new= L1 + 10*times
        L2_new= L2 + 10*times

        print("-------------------------------times:",times,"new L1:", L1_new,"new L2:", L2_new)
        roc_address1 = roc_address + "_" +str(times) + ".pdf"
        y_scores_mean,model = cross_validation_by_model_3(n_splits,save_model_address,model_types, X_train, Y_train, X_train, Y_train, image_size1, image_size2, image_size3, label_types, epochs, times, L1_new, L2_new, F1, F2, F3, batch_size, roc_address1, roc_title)
        if y_scores_mean > y_scores_max: 
            times_best = times
            y_scores_max = y_scores_mean
        print("$$$$$$$$$$$$$$$$$$$$$$y_scores_mean:", y_scores_mean, "  y_scores_max:", y_scores_max, "  times:",times, "  times_best:", times_best)
    ####
    L1_best=L1 + 10 * times 
    L2_best=L2 + 10 * times  
    print(" L1_best:", L1_best, " L2_best:", L2_best)
    #print("Y_train",Y_train)    
    #model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1_best, L2_best, F1, F2, F3)
    #model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.001, patience=5, verbose=2)
    model=create_model(model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1_best, L2=L2_best, F1=F1, F2=F2, F3=F3)
    #model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2, image_size3=image_size3, label_types=label_types, times=times, L1=L1_best, L2=L2_best, F1=F1, F2=F2, F3=F3, epochs=epochs, batch_size=2, verbose=1)
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
    print("before fit ---------Y_train:",Y_train)
    model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_train, Y_train), epochs=epochs,verbose=2,  shuffle=False, callbacks=[early_stopping])
    probas_ = model.predict(X_train)
    print("after prodict-----probas_:",probas_, "label_types: ",label_types)
    fpr, tpr, thresholds = roc_curve(Y_train[:,1], probas_[:, 1])
    
    ####

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    ##https://mlfromscratch.com/gridsearch-keras-sklearn/#/
    ##GridSearchCV – you might wonder why 'neg_log_loss' was used as the scoring method?
    ##The solution to using something else than negative log loss is to remove some of the preprocessing 
    ##of the MNIST dataset; that is, REMOVE the part where we make the output variables categorical
    ## Categorical y values
    ## y_train = to_categorical(y_train)
    ## y_test= to_categorical(y_test)


    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred

def cross_validation_select_parameters_4(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title):
    print("begin of crosss_validation_select_parameters_4: ")
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    #Y_train = to_categorical(Y_train, num_classes=None)
    #######
    #from xgboost import XGBClassifier                     #xgboost
 
    seed = 7 #重现随机生成的训练
    #test_size = 0.33 #33%测试，67%训练
    test_size = 0.50 #33%测试，67%训练
    X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_labels, test_size=test_size, random_state=seed)
    ###
    param_grid = {
              'epochs':[10],
              "times":[2],
              #'fs1':[3,5]
              #'epochs' :              [100,150,200],
              #'batch_size' :          [32, 128],
              #'optimizer' :           ['Adam', 'Nadam'],
              #'dropout_rate' :        [0.2, 0.3]
              #'activation' :          ['relu', 'elu']
              #'init_mode' :['uniform', 'lecun_uniform', 'normal']
             }
    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #init_mode = ['uniform','lecun_uniform','normal']
    #param_grid = dict(init_mode=init_mode)
    #model = XGBClassifier()               
    model = KerasClassifier(build_fn = build_cnn,activation = 'relu', dropout_rate = 0.2, optimizer = 'Adam', fs1 = 5, times = times, init_mode='uniform', verbose=1)
    
    #model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2, image_size3=image_size3, label_types=label_types, times=times, L1=L1, L2=L2, F1=F1, F2=F2, F3=F3, epochs=epochs, batch_size=2, verbose=1)
    model1, pred = algorithm_pipeline(X_train, X_test, Y_train, Y_test, model, 
                                        param_grid, cv=2, scoring_fit='neg_log_loss')

    print("model1.best_score_:",model1.best_score_)
    print("model1.best_params_:",model1.best_params_)
    #print("model1.best_estimator_:",model1.best_estimator_)
    print("model1.cv_results_:",model1.cv_results_)
    print("pred:",pred)

    ###
    #times_select = [0.0001,0.001,0.01,0.1,0.2,0.3] #学习率
    #L1_select = [1, 0.1, 0.01, 0.001]
    #param_grid = dict(L1 = learning_rate,gamma = gamma)#转化为字典格式，网络搜索要求
    #kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)#将训练/测试数据集划分10个互斥子集，
    #grid_search = GridSearchCV(model,param_grid,scoring = 'neg_log_loss',n_jobs = -1,cv = kflod)
    #scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
    #grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
    #print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
    #grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
    #best_score_：成员提供优化过程期间观察到的最好的评分
    #具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
    #注意，“params”键用于存储所有参数候选项的参数设置列表。
    #means = grid_result.cv_results_['mean_test_score']
    #params = grid_result.cv_results_['params']
    #for mean,param in zip(means,params):
    #    print("%f  with:   %r" % (mean,param))
 
    ######    

def model_train(save_model_address,model_types,train_images,train_labels, image_size1, image_size2, image_size3, label_types, epochs,times, L1, L2, F1, F2, F3, n_splits, cross_valadation_code=1, batch_size=2, roc_address="roc.pdf", roc_title="ROC curve"):
    print("-----------------------image_size1:",image_size1,"image_size2:",image_size2)
    print("model_types:", model_types)
    print("in the begining of model_train:",save_model_address)
    tf.config.threading.set_inter_op_parallelism_threads = 50
    #tf.config.threading.set_intra_op_parallelism_threads = 30
    #tf.config.device_count = 5
    #print("tf.config.device_count:",tf.config.device_count)
    # 只使用一个线程
    os.environ["OMP_NUM_THREADS"] = "1"
    if(cross_valadation_code==1):
        cross_validation_1(n_splits,save_model_address,model_types,train_images,train_labels,train_images, train_labels, image_size1, image_size2, image_size3,label_types,epochs,times,L1,L2,F1,F2,F3,batch_size,roc_address)
    elif(cross_valadation_code==2):
        Keras_Classifier(n_splits,save_model_address,model_types, train_images, train_labels, train_images, train_labels, image_size1, image_size2, image_size3,label_types, epochs, times, L1, L2, F1, F2, F3)
    elif(cross_valadation_code==3):
        cross_validation_select_parameters_3(n_splits,save_model_address,model_types, train_images, train_labels, train_images, train_labels, image_size2, image_size2, image_size3,label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title)
    elif(cross_valadation_code==4):
        cross_validation_select_parameters_4(n_splits,save_model_address,model_types, train_images, train_labels, train_images, train_labels, image_size2, image_size2, image_size3,label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title)
    elif(cross_valadation_code==0):
        train_by_all(n_splits,save_model_address,model_types,train_images,train_labels,train_images, train_labels, image_size1, image_size2, image_size3,label_types,epochs,times,L1,L2,F1,F2,F3,batch_size,roc_address,roc_title)

    '''
    #
    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit( train_images, train_labels)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
       print("%f (%f) with: %r" % (mean, stdev, param))
    '''
def model_test(save_model_address, model_name, test_images, test_labels,roc_address,roc_title="ROC curve"):

    # step8
    #  returns a compiled model
    #  identical to the previous one
    restored_model = tf.keras.models.load_model(save_model_address)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_labels = to_categorical(test_labels, num_classes=None)
    # step9
    loss, acc = restored_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
    # pred = restored_model.predict(test_images[:2])
    #print('predict:', pred )
    # https://blog.csdn.net/wwwlyj123321/article/details/94291992
    # 利用model.predict获取测试集的预测值
    y_pred = restored_model.predict(test_images, batch_size=1)
    #draw_roc_curve(test_labels[:,1], y_pred[:,1], roc_address, roc_title)
    auc_title = ""
    draw_roc.draw_roc_curve(Y_test[:,1], probas_[:, 1],roc_address,auc_title,roc_title=roc_title)

def model_test_single(save_model_address, test_image):

    # step8
    #  returns a compiled model
    #  identical to the previous one
    restored_model = tf.keras.models.load_model(save_model_address)
    
    test_image = np.array(test_image)
    # step9
    # pred = restored_model.predict(test_images[:2])
    #print('predict:', pred )
    # https://blog.csdn.net/wwwlyj123321/article/details/94291992
    # 利用model.predict获取测试集的预测值
    y_pred = restored_model.predict(test_image, batch_size=1)
    return y_pred





