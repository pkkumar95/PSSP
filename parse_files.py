'''
This file is part of HHMFold, for the paper 'Advancing the Accuracy of Protein Fold
Recognition by Utilizing Profiles from Hidden Markov Models'
Code written by James Lyons, December 2014
This file is released under an MIT licence

parse_files.py - provides routines for parsing hmm files, pssm files and the label files
used in the paper for the TG, EDD and DD protein fold recognition datasets
'''
import numpy

'''
load the labels for a database. 
label_file - a string containing the path to the text file containing 2 columns, the 
  first column is the label (an integer), second column is the protein sequence (string).
'''
def load_labels(label_file):
    f = open(label_file)
    labels = {}
    for line in f:
        toks = line.split()
        labels[toks[1].upper()] = int(toks[0])
    return labels

'''
parse a HHBLITS hmm file and return the contents.
fname - a string containing the path to the hmm text file output by HHBLITS
returns:
seq - the protein sequence
prob - the linear probabilties
extras - ten probabilites for each state about insertions, deletions, etc.
'''
def parse_hmm(fname):
    f = open(fname)
    line=f.readline()
    while line[0]!='#':
        line=f.readline()   
    f.readline()
    f.readline()
    f.readline()
    f.readline()    
    seq = []
    extras = numpy.zeros([0,10])
    prob = numpy.zeros([0,20])
    line = f.readline()
    while line[0:2]!='//':
        lineinfo = line.split()
        seq.append(lineinfo[0])  
        probs_ = [2**(-float(lineinfo[i])/1000) if lineinfo[i]!='*' else 0. for i in range(2,22)]
        prob = numpy.concatenate((prob,numpy.matrix(probs_)),axis=0)
        
        line = f.readline()
        lineinfo = line.split()
        extras_ = [2**(-float(lineinfo[i])/1000) if lineinfo[i]!='*' else 0. for i in range(0,10)]
        extras = numpy.concatenate((extras,numpy.matrix(extras_)),axis=0)
        
        line = f.readline()
        assert len(line.strip())==0
        
        line = f.readline()
    return (''.join(seq),prob,extras)

'''
parse a PSIBLAST pssm file and return the contents.
fname - a string containing the path to the pssm text file output by PSIBLAST
returns:
seq - the protein sequence
prob - the linear probabilties, divided by 100
lprob - the log probabilities
'''
def parse_pssm(fname):
    f = open(fname)
    # the 4th line should be the start of the PSSM data
    f.readline()
    f.readline()
    f.readline()
    seq = []
    lprob = numpy.zeros([0,20])
    prob = numpy.zeros([0,20])
    line = f.readline()
    while len(line.strip())>0:
        lineinfo = line.split()
        seq.append(lineinfo[1])
        lprobs_ = [float(lineinfo[i]) for i in range(2,22)]
        lprob = numpy.concatenate((lprob,numpy.matrix(lprobs_)),axis=0)
        probs_ = [float(lineinfo[i])/100 for i in range(22,42)]
        prob = numpy.concatenate((prob,numpy.matrix(probs_)),axis=0)
        line = f.readline()
    
    return (''.join(seq),prob,lprob)
