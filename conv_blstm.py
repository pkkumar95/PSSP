import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import Activation, Reshape, TimeDistributed, Embedding, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from keras import backend as K
import numpy as np
import pickle
import csv


def accuracy(y_true, y_pred):
    count = 0
    length = 0
    acc = 0
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if np.sum(y_true[i,j,:]) != 0:
                length = length + 1
                if np.argmax(y_true[i,j,:]) == np.argmax(y_pred[i,j,:]):
                    count = count + 1
    acc = count / length
    return acc
    
def weighted_accuracy(y_true, y_pred):
    return K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), dtype='float32') * K.sum(y_true, axis=-1)) / K.sum(y_true)  

trainhmm_6133 = pickle.load(open('./pickles/trainhmm_cull_6133.pickle', 'rb'))
trainlabel_6133 = pickle.load(open('./pickles/trainlabel_cull_6133.pickle', 'rb'))
trainpssm_6133 = pickle.load(open('./pickles/trainpssm_cull_6133.pickle', 'rb'))

valhmm_6133 = pickle.load(open('./pickles/valhmm_cull_6133.pickle', 'rb'))
valpssm_6133 = pickle.load(open('./pickles/valpssm_cull_6133.pickle', 'rb'))
vallabel_6133 = pickle.load(open('./pickles/vallabel_cull_6133.pickle', 'rb'))

testhmm_6133 = pickle.load(open('./pickles/testhmm_cull_6133.pickle', 'rb'))
testpssm_6133 = pickle.load(open('./pickles/testpssm_cull_6133.pickle', 'rb'))
testlabel_6133 = pickle.load(open('./pickles/testlabel_cull_6133.pickle', 'rb'))

testhmm_513 = pickle.load(open('./pickles/testhmm_cb513.pickle', 'rb'))
testpssm_513 = pickle.load(open('./pickles/testpssm_cb513.pickle', 'rb'))
testlabel_513 = pickle.load(open('./pickles/testlabel_cb513.pickle', 'rb'))

testhmm_10 = pickle.load(open('./pickles/testhmm_casp10.pickle', 'rb'))
testpssm_10 = pickle.load(open('./pickles/testpssm_casp10.pickle', 'rb'))
testlabel_10 = pickle.load(open('./pickles/testlabel_casp10.pickle', 'rb'))

testhmm_11 = pickle.load(open('./pickles/testhmm_casp11.pickle', 'rb'))
testpssm_11 = pickle.load(open('./pickles/testpssm_casp11.pickle', 'rb'))
testlabel_11 = pickle.load(open('./pickles/testlabel_casp11.pickle', 'rb'))

with tf.device('/device:GPU:1'):
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    K.tensorflow_backend.set_session(tf.Session(config=config))
    
    layer_size = 42
    
    #main_input = Input(shape=(700,20), name='main_input')
    #main_input = Embedding(output_dim=1, input_dim=20, input_length=700)(main_input)
    main_input = Input(shape=(700,21), name='main_input')
    aux_input = Input(shape=(700,21), name='aux_input')
    input_features = concatenate([main_input, aux_input], axis=-1, name='c1')
    
    c_input = Reshape((700,42,1))(input_features)
    
    #One layer of Convolutional 2D 
    
    c_output = Conv2D(layer_size, (3,3), activation='relu', padding='same', bias_regularizer=l2(0.001))(c_input)
    print('c_output: ', c_output.get_shape())
    
    m_output = Reshape((700,42*42))(c_output)
    m_output = Dropout(0.2)(m_output)
    d_output = Dense(400, activation='relu')(m_output)
    
    #Bidirectional RNN with LSTM cells
    f1 = LSTM(250, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2)(d_output)
    f2 = LSTM(250, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, go_backwards=True)(d_output)
    
    f3 = LSTM(250, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2)(f1)
    f4 = LSTM(250, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, go_backwards=True)(f2)
    
    cf_feature = concatenate([f3,f4, m_output], axis=2, name='concatenate')
    cf_feature = Dropout(0.2)(cf_feature)
    
    f_input = Dense(600, activation='relu')(cf_feature)
    
    main_output = TimeDistributed(Dense(3, activation='softmax'), name='main_output')(f_input)
    
    model = Model(inputs=[main_input, aux_input], outputs=[main_output])
    adam = Adam(lr=0.003)
    
    model.compile(optimizer=adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    
    model.summary()
    
    earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
    load_file = "./model/UC_Q3_conv_blstm.h5"
    checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)
    
    history=model.fit({'main_input': trainpssm_6133, 'aux_input': trainhmm_6133},
              {'main_output': trainlabel_6133},validation_data=({'main_input': valpssm_6133, 'aux_input': valhmm_6133},{'main_output': vallabel_6133}),
              epochs=200, batch_size=64, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)
    
    
    model.load_weights(load_file)
    
    print("#########evaluate UC_Q3_conv_blstm:##############")
    y_pred_6133 = model.predict({'main_input': testpssm_6133, 'aux_input': testhmm_6133},verbose=1,batch_size=1)
    score_6133 = model.evaluate({'main_input': testpssm_6133, 'aux_input': testhmm_6133},{'main_output': testlabel_6133},verbose=1,batch_size=1)
    print(score_6133)
    print('test loss 6133:', score_6133[0])
    print('test accuracy 6133:', score_6133[1])
    acc_6133 = accuracy(testlabel_6133, y_pred_6133)
    #print('length_6133: ', length_6133)
    print('Q3_accuracy 6133: ', acc_6133)
    with open('./results_conv_blstm/cb6133/cb6133_pred.csv', 'w+') as file:
        fields = ['pdb_id', 'y_true', 'y_pred']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        seq_t = str()
        seq_p = str()
        for i in range(testlabel_6133.shape[0]):
            row = dict()
            pdb = str(i+1)
            # l = 0
            for j in range(testlabel_6133.shape[1]):
                if np.sum(testlabel_6133[i,j,:]) != 0:
                    yt = int(np.argmax(testlabel_6133[i,j,:]))
                    yp = int(np.argmax(y_pred_6133[i,j,:]))
                    seq_t = seq_t + str(yt)
                    seq_p = seq_p + str(yp)
                    # l = l + 1
            row['pdb_id'] = pdb
            row['y_true'] = seq_t
            row['y_pred'] = seq_p
            writer.writerow(row)
            # print(l)
            with open('./results_conv_blstm/cb6133/in/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_t)
            fasta.close()
            with open('./results_conv_blstm/cb6133/out/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_p)
            fasta.close()
            row = dict()
            pdb = str()
            seq_t = str()
            seq_p = str()
    file.close()         
    
    y_pred_cb513 = model.predict({'main_input': testpssm_513, 'aux_input': testhmm_513},verbose=1,batch_size=1)
    score_cb513 = model.evaluate({'main_input': testpssm_513, 'aux_input': testhmm_513},{'main_output': testlabel_513},verbose=1,batch_size=1)
    print(score_cb513)
    print('test loss cb513:', score_cb513[0])
    print('test accuracy cb513:', score_cb513[1])
    acc_cb513 = accuracy(testlabel_513, y_pred_cb513)
    print('Q3_accuracy cb513: ', acc_cb513)
    with open('./results_conv_blstm/cb513/cb513_pred.csv', 'w+') as file:
        fields = ['pdb_id', 'y_true', 'y_pred']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        seq_t = str()
        seq_p = str()
        for i in range(testlabel_513.shape[0]):
            pdb = str(i+1)
            row = dict()
            for j in range(testlabel_513.shape[1]):
                if np.sum(testlabel_513[i,j,:]) != 0:
                    yt = int(np.argmax(testlabel_513[i,j,:]))
                    yp = int(np.argmax(y_pred_cb513[i,j,:]))
                    seq_t = seq_t + str(yt)
                    seq_p = seq_p + str(yp)
            row['pdb_id'] = pdb
            row['y_true'] = seq_t
            row['y_pred'] = seq_p
            writer.writerow(row)
            with open('./results_conv_blstm/cb513/in/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_t)
            fasta.close()
            with open('./results_conv_blstm/cb513/out/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_p)
            fasta.close()
            row = dict()
            pdb = str()
            seq_t = str()
            seq_p = str()
    file.close()
    
    y_pred_casp10 = model.predict({'main_input': testpssm_10, 'aux_input': testhmm_10},verbose=1,batch_size=1)
    score_casp10 = model.evaluate({'main_input': testpssm_10, 'aux_input': testhmm_10},{'main_output': testlabel_10},verbose=1,batch_size=1)
    print(score_casp10)
    print('test loss casp10:', score_casp10[0])
    print('test accuracy casp10:', score_casp10[1])
    acc_casp10 = accuracy(testlabel_10, y_pred_casp10)
    print('Q3_accuracy casp10: ', acc_casp10)
    with open('./results_conv_blstm/casp10/casp10_pred.csv', 'w+') as file:
        fields = ['pdb_id', 'y_true', 'y_pred']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        seq_t = str()
        seq_p = str()
        for i in range(testlabel_10.shape[0]):
            pdb = str(i+1)
            row = dict()
            for j in range(testlabel_10.shape[1]):
                if np.sum(testlabel_10[i,j,:]) != 0:
                    yt = int(np.argmax(testlabel_10[i,j,:]))
                    yp = int(np.argmax(y_pred_casp10[i,j,:]))
                    seq_t = seq_t + str(yt)
                    seq_p = seq_p + str(yp)
            row['pdb_id'] = pdb
            row['y_true'] = seq_t
            row['y_pred'] = seq_p
            writer.writerow(row)
            with open('./results_conv_blstm/casp10/in/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_t)
            fasta.close()
            with open('./results_conv_blstm/casp10/out/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_p)
            fasta.close()
            row = dict()
            pdb = str()
            seq_t = str()
            seq_p = str()
    file.close()
    
    y_pred_casp11 = model.predict({'main_input': testpssm_11, 'aux_input': testhmm_11},verbose=1,batch_size=1)
    score_casp11 = model.evaluate({'main_input': testpssm_11, 'aux_input': testhmm_11},{'main_output': testlabel_11},verbose=1,batch_size=1)
    print(score_casp11)
    print('test loss casp11:', score_casp11[0])
    print('test accuracy casp11:', score_casp11[1])
    acc_casp11 = accuracy(testlabel_11, y_pred_casp11)
    print('Q3_accuracy casp11: ', acc_casp11)
    with open('./results_conv_blstm/casp11/casp11_pred.csv', 'w+') as file:
        fields = ['pdb_id', 'y_true', 'y_pred']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        seq_t = str()
        seq_p = str()
        for i in range(testlabel_11.shape[0]):
            pdb = str(i+1)
            row = dict()
            for j in range(testlabel_11.shape[1]):
                if np.sum(testlabel_11[i,j,:]) != 0:
                    yt = int(np.argmax(testlabel_11[i,j,:]))
                    yp = int(np.argmax(y_pred_casp11[i,j,:]))
                    seq_t = seq_t + str(yt)
                    seq_p = seq_p + str(yp)
            row['pdb_id'] = pdb
            row['y_true'] = seq_t
            row['y_pred'] = seq_p
            writer.writerow(row)
            with open('./results_conv_blstm/casp11/in/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_t)
            fasta.close()
            with open('./results_conv_blstm/casp11/out/pdb_'+pdb+'.fasta', 'w+') as fasta:
                fasta.write('>pdb_'+pdb+'\n')
                fasta.write(seq_p)
            fasta.close()
            row = dict()
            pdb = str()
            seq_t = str()
            seq_p = str()
    file.close()
