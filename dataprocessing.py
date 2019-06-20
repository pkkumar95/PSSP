
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import pickle

import parse_files as p

# In[2]:


def load_cull6133():

    hmm = []
    for i in range(1,6134):
        #print(i)
        seq,prob,extras = p.parse_hmm("./data/cb6133/hhm/pdbid" + str(i) + ".txt")
        norm = np.concatenate((prob,extras[:,7]),axis=1)
        norm += 0.05
        if len(norm) < 700:
            for j in range(700-len(norm)):
                norm = np.concatenate((norm,norm[0]*0))
        else:
            norm = norm[:700]
        hmm.append(norm)
    hmm = np.array(hmm)
    

    cull = np.load("./data/cullpdb+profile_6133.npy")
    cull = np.reshape(cull, (-1, 700, 57))
    
    

    labels = cull[:, :, 22:30]
    onehot = cull[:, :, 0:21]
    pssm = cull[:, :, 35:56]
    
    #print(onehot.shape)
    
    num_seqs, seqlen, feature_dim = np.shape(cull)[0], np.shape(cull)[1], 78
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)
    np.random.shuffle(seq_index)
#     print(seq_index, num_seqs, seqlen, feature_dim, num_classes)
    
    print("Loading train/dev data (cullpdb+profile_6133)...")
    trainhot = onehot[seq_index[:5600]]
    #print(trainhot.shape)
    trainlabel = labels[seq_index[:5600]]
    trainpssm = pssm[seq_index[:5600]]   
#	plus HMM
    trainhmm = hmm[seq_index[:5600]]
    
    vallabel = labels[seq_index[5877:6133]]
    valpssm = pssm[seq_index[5877:6133]]
# 	plus HMM
    valhmm = hmm[seq_index[5877:6133]]
    valhot = onehot[seq_index[5877:6133]]
    #print(valhot.shape)
    
    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    #print(trainhot[0][1])
    #print(train_hot[1])
    for i in range(trainhot.shape[0]):
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i,j,:]) != 0:
                train_hot[i,j] = np.argmax(trainhot[i,j,:])
            else:
                train_hot[i,j] = 21
#     print(train_hot[0])

    
    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in range(valhot.shape[0]):
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i,j,:]) != 0:
                val_hot[i,j] = np.argmax(valhot[i,j,:])
            else:
                val_hot[i,j] = 21
                
    
    train_lab = np.zeros((trainlabel.shape[0], trainlabel.shape[1], 3))

    helix = [3,4,5]
    strand = [1,2]
    coil = [0,6,7]

    for i in range(trainlabel.shape[0]):
        for j in range(trainlabel.shape[1]):
            if np.sum(trainlabel[i,j,:]) != 0:
                if int(np.argmax(trainlabel[i,j,:])) in helix:
                    train_lab[i,j,0] = 1
                elif int(np.argmax(trainlabel[i,j,:])) in strand:
                    train_lab[i,j,1] = 1
                else:
                    train_lab[i,j,2] = 1
    
    val_lab = np.zeros((vallabel.shape[0], vallabel.shape[1], 3))
    
    for i in range(vallabel.shape[0]):
        for j in range(vallabel.shape[1]):
            if np.sum(vallabel[i,j,:]) != 0:
                if int(np.argmax(vallabel[i,j,:])) in helix:
                    val_lab[i,j,0] = 1
                elif int(np.argmax(vallabel[i,j,:])) in strand:
                    val_lab[i,j,1] = 1
                else:
                    val_lab[i,j,2] = 1
    
    
    print("Loading Test data (cullpdb+profile_6133)...")
    
    testhot = onehot[seq_index[5605:5877]]
    #print (testhot.shape)
    testlabel = labels[seq_index[5605:5877]]
    testpssm = pssm[seq_index[5605:5877]]
#	plus HMM
    testhmm = hmm[seq_index[5605:5877]]
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
            else:
                test_hot[i,j] = 21
    
    test_lab = np.zeros((testlabel.shape[0], testlabel.shape[1], 3))
    for i in range(testlabel.shape[0]):
        for j in range(testlabel.shape[1]):
            if np.sum(testlabel[i,j,:]) != 0:
                if int(np.argmax(testlabel[i,j,:])) in helix:
                    test_lab[i,j,0] = 1
                elif int(np.argmax(testlabel[i,j,:])) in strand:
                    test_lab[i,j,1] = 1
                else:
                    test_lab[i,j,2] = 1
    
    
    pickle_out = open("./pickles/trainhot_cull_6133.pickle", "wb")
    pickle.dump(train_hot, pickle_out)
    pickle_out.close()
    
    pickle_out = open("./pickles/trainpssm_cull_6133.pickle", "wb")
    pickle.dump(trainpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/trainlabel_cull_6133.pickle", "wb")
    pickle.dump(train_lab, pickle_out)
    pickle_out.close()
#	Plus HMM
    pickle_out = open("./pickles/trainhmm_cull_6133.pickle", "wb")
    pickle.dump(trainhmm, pickle_out)
    pickle_out.close()
    

    pickle_out = open("./pickles/valhot_cull_6133.pickle", "wb")
    pickle.dump(val_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/valpssm_cull_6133.pickle", "wb")
    pickle.dump(valpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/vallabel_cull_6133.pickle", "wb")
    pickle.dump(val_lab, pickle_out)
    pickle_out.close()
    
#	Plus HMM
    pickle_out = open("./pickles/valhmm_cull_6133.pickle", "wb")
    pickle.dump(valhmm, pickle_out)
    pickle_out.close()


    pickle_out = open("./pickles/testhot_cull_6133.pickle", "wb")
    pickle.dump(test_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testpssm_cull_6133.pickle", "wb")
    pickle.dump(testpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testlabel_cull_6133.pickle", "wb")
    pickle.dump(test_lab, pickle_out)
    pickle_out.close()
#	Plus HMM
    pickle_out = open("./pickles/testhmm_cull_6133.pickle", "wb")
    pickle.dump(testhmm, pickle_out)
    pickle_out.close()


    #print(train_hot.shape, trainpssm.shape, trainlabel.shape,trainhmm.shape)
   # print(val_hot.shape, valpssm.shape, vallabel.shape,valhmm.shape) 
    #print(test_hot.shape, testpssm.shape, testlabel.shape,testhmm.shape)
    
    # return trainhot, trainpssm, trainlabel, valhot, valpssm, vallabel, testhot, testpssm, testlabel


# In[3]:


load_cull6133()


# In[4]:





# In[5]:


'''def load_cull6133_filtered():

    hmm = []
    for i in range(1,6134):
        print(i)
        seq,prob,extras = p.parse_hmm("/home/15it103/MajorProject/HHBlits/Datasets/cb6133/hhm/pdbid" + str(i) + ".txt")
        norm = np.concatenate((prob,extras[:,7]),axis=1)
        norm += 0.05
        if len(norm) < 700:
            for j in range(700-len(norm)):
                norm = np.concatenate((norm,norm[0]*0))
        else:
            norm = norm[:700]
        hmm.append(norm)
    hmm = np.array(hmm)
    

    data = np.load("../JBCB2018/jbcb/data/cullpdb+profile_6133_filtered.npy")
    data = np.reshape(data, (-1, 700, 57))
    
    datahot = data[:, :, 0:21]
    labels = data[:, :, 22:30]
    datapssm = data[:, :, 35:56]
    
    num_seqs, seqlen, feature_dim = np.shape(data)[0], np.shape(data)[1], 78
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)
    np.random.shuffle(seq_index)
    
    #train data
    trainhot = datahot[seq_index[:5278]]
    trainlabel = labels[seq_index[:5278]]
    trainpssm = datapssm[seq_index[:5278]]
    trainhmm = hmm[seq_index[:5278]]
    
    #val data
    vallabel = labels[seq_index[5278:5534]]
    valpssm = datapssm[seq_index[5278:5534]]
    valhot = datahot[seq_index[5278:5534]]
    valhmm = hmm[seq_index[5278:5534]]
    
    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    for i in range(trainhot.shape[0]):
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i,j,:]) != 0:
                train_hot[i,j] = np.argmax(trainhot[i,j,:])
            else:
                train_hot[i,j] = 21
    
    
    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in range(valhot.shape[0]):
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i,j,:]) != 0:
                val_hot[i,j] = np.argmax(valhot[i,j,:])
            else:
                val_hot[i,j] = 21
                
    print(datahot.shape, trainhot.shape, valhot.shape, trainhot.shape[0]+valhot.shape[0])
    
    pickle_out = open("./pickles2/trainhot_cull_6133_filtered.pickle", "wb")
    pickle.dump(train_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/trainpssm_cull_6133_filtered.pickle", "wb")
    pickle.dump(trainpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/trainlabel_cull_6133_filtered.pickle", "wb")
    pickle.dump(trainlabel, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/trainhmm_cull_6133_filtered.pickle", "wb")
    pickle.dump(trainhmm, pickle_out)
    pickle_out.close()


    pickle_out = open("./pickles2/valhot_cull_6133_filtered.pickle", "wb")
    pickle.dump(val_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/valpssm_cull_6133_filtered.pickle", "wb")
    pickle.dump(valpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/vallabel_cull_6133_filtered.pickle", "wb")
    pickle.dump(vallabel, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles2/valhmm_cull_6133_filtered.pickle", "wb")
    pickle.dump(valhmm, pickle_out)
    pickle_out.close()

    
    # return trainhot, trainpssm, trainlabel, valhot, valpssm, vallabel


# In[6]:


load_cull6133_filtered()
'''

# In[7]:


def load_cb513():
    print ("Loading Test data (CB513)...")

    hmm = []
    for i in range(1,515):
        #print(i)
        seq,prob,extras = p.parse_hmm("./data/cb513/hhm/pdbid" + str(i) + ".txt")
        norm = np.concatenate((prob,extras[:,7]),axis=1)
        norm += 0.05
        if len(norm) < 700:
            for j in range(700-len(norm)):
                norm = np.concatenate((norm,norm[0]*0))
        else:
            norm = norm[:700]
        hmm.append(norm)
    hmm = np.array(hmm)
    

    cb513 = np.load("./data/cb513+profile_split1.npy")
    cb513 = np.reshape(cb513, (-1, 700, 57))
    
    datahot = cb513[:, :, 0:21]
    labels = cb513[:, :, 22:30]
    datapssm = cb513[:, :, 35:56]
    
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    testhmm =  hmm
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
            else:
                test_hot[i,j] = 21
    
    test_lab = np.zeros((testlabel.shape[0], testlabel.shape[1], 3))
    helix = [3,4,5]
    strand = [1,2]
    coil = [0,6,7]
    for i in range(testlabel.shape[0]):
        for j in range(testlabel.shape[1]):
            if np.sum(testlabel[i,j,:]) != 0:
                if int(np.argmax(testlabel[i,j,:])) in helix:
                    test_lab[i,j,0] = 1
                elif int(np.argmax(testlabel[i,j,:])) in strand:
                    test_lab[i,j,1] = 1
                else:
                    test_lab[i,j,2] = 1
    
    
   # print(testhot.shape)
    
    pickle_out = open("./pickles/testhot_cb513.pickle", "wb")
    pickle.dump(test_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testpssm_cb513.pickle", "wb")
    pickle.dump(testpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testlabel_cb513.pickle", "wb")
    pickle.dump(test_lab, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testhmm_cb513.pickle", "wb")
    pickle.dump(testhmm, pickle_out)
    pickle_out.close()

    
    # return testhot, testpssm, testlabel


# In[8]:


load_cb513()


# In[9]:


def load_casp10():
    print ("Loading Test data (CASP10)...")

    hmm = []
    for i in range(1,124):
        #print(i)
        seq,prob,extras = p.parse_hmm("./data/casp10/hhm/pdbid" + str(i) + ".txt")
        norm = np.concatenate((prob,extras[:,7]),axis=1)
        norm += 0.05
        if len(norm) < 700:
            for j in range(700-len(norm)):
                norm = np.concatenate((norm,norm[0]*0))
        else:
            norm = norm[:700]
        hmm.append(norm)
    hmm = np.array(hmm)
    
    casp10 = h5py.File("./data/casp10.h5")
    
    datahot = casp10['features'][:, :, 0:21]
    datapssm = casp10['features'][:, :, 21:42]
    labels = casp10['labels'][:, :, 0:8]
    
    testhot = datahot
    testpssm = datapssm
    testlabel = labels
    testhmm = hmm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
            else:
                test_hot[i,j] = 21
                
    test_lab = np.zeros((testlabel.shape[0], testlabel.shape[1], 3))
    helix = [3,4,5]
    strand = [1,2]
    coil = [0,6,7]
    for i in range(testlabel.shape[0]):
        for j in range(testlabel.shape[1]):
            if np.sum(testlabel[i,j,:]) != 0:
                if int(np.argmax(testlabel[i,j,:])) in helix:
                    test_lab[i,j,0] = 1
                elif int(np.argmax(testlabel[i,j,:])) in strand:
                    test_lab[i,j,1] = 1
                else:
                    test_lab[i,j,2] = 1
    
    
    pickle_out = open("./pickles/testhot_casp10.pickle", "wb")
    pickle.dump(test_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testpssm_casp10.pickle", "wb")
    pickle.dump(testpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testlabel_casp10.pickle", "wb")
    pickle.dump(test_lab, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testhmm_casp10.pickle", "wb")
    pickle.dump(testhmm, pickle_out)
    pickle_out.close()

    
    # return testhot, testpssm, testlabel


# In[10]:


load_casp10()


# In[11]:


def load_casp11():
    print ("Loading Test data (CASP11)...")

    hmm = []
    for i in range(1,106):
       # print(i)
        seq,prob,extras = p.parse_hmm("./data/casp11/hhm/pdbid" + str(i) + ".txt")
        norm = np.concatenate((prob,extras[:,7]),axis=1)
        norm += 0.05
        if len(norm) < 700:
            for j in range(700-len(norm)):
                norm = np.concatenate((norm,norm[0]*0))
        else:
            norm = norm[:700]
        hmm.append(norm)
    hmm = np.array(hmm)
    

    casp11 = h5py.File("./data/casp11.h5")
    
    datahot=casp11['features'][:, :, 0:21]
    datapssm=casp11['features'][:, :, 21:42]
#    datapssm = hmm
    labels = casp11['labels'][:, :, 0:8]
    
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    testhmm = hmm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
            else:
                test_hot[i,j] = 21
    
    test_lab = np.zeros((testlabel.shape[0], testlabel.shape[1], 3))
    helix = [3,4,5]
    strand = [1,2]
    coil = [0,6,7]
    for i in range(testlabel.shape[0]):
        for j in range(testlabel.shape[1]):
            if np.sum(testlabel[i,j,:]) != 0:
                if int(np.argmax(testlabel[i,j,:])) in helix:
                    test_lab[i,j,0] = 1
                elif int(np.argmax(testlabel[i,j,:])) in strand:
                    test_lab[i,j,1] = 1
                else:
                    test_lab[i,j,2] = 1
    
    pickle_out = open("./pickles/testhot_casp11.pickle", "wb")
    pickle.dump(test_hot, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testpssm_casp11.pickle", "wb")
    pickle.dump(testpssm, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testlabel_casp11.pickle", "wb")
    pickle.dump(test_lab, pickle_out)
    pickle_out.close()
    pickle_out = open("./pickles/testhmm_casp11.pickle", "wb")
    pickle.dump(testhmm, pickle_out)
    pickle_out.close()

    
    # return testhot, testpssm, testlabel


# In[12]:


load_casp11()

