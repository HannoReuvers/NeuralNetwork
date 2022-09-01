#!/usr/local/bin/python3

import numpy as np
import os
from pathlib import Path
import sqlite3

#---------------#
#--- GENERAL ---#
#---------------#
# Activation functions
sigmoid = lambda x: 1/(1+np.exp(-x))
softmax = lambda x: np.exp(x)/np.sum( np.exp(x), axis=0, keepdims=True)
Tanh = lambda x: np.tanh(x)
ReLU = lambda x: x*(x>0)

#----------------------------#
#--- NEURAL NETWORK CLASS ---#
#----------------------------#
class NeuralNetwork:

    # Constants
    MaxEpochs = 20000

    # Activation function (None signifies to transformation to input)
    fList = [None, Tanh, Tanh, softmax]

    # Initialize empty lists
    X = {}
    y = {}
    yOneHot = {}
    a = {}
    b = {}
    cost = {}
    W = {}
    z = {}

    def __init__(self, NeuronsList):
        '''Initialize all neural network variables: numpy arrays start with NaN default'''
        self.NeuronsList = NeuronsList
        self.NumberOfLayers = len(self.NeuronsList)-1

    def __str__(self):
        ''' Print summary of neural network layout'''
        return "Fully connected network with {} layer(s) and {}".format(self.NumberOfLayers, self.NeuronsList)

    def ReadHRInitialsData(self, DataType):
        '''Read data from SQL file'''

        # Baseline folder
        path = Path('.')
        SQLpath = path.parent.absolute().parent.absolute().joinpath('HRInitialsClassificationData')

        # Select file to read
        if DataType in ['train', 'valid', 'test']:
            sqlfilename = 'HR_'+DataType+'.sqlite'
        else:
            raise ValueError('Data specification should be train, valid or test.')

        # Read data using SQLite
        conn = sqlite3.connect(str(SQLpath)+'/'+sqlfilename)
        cur = conn.cursor()
        cur.execute('SELECT * FROM Data')
        data = cur.fetchall()

        # X, y, and yOneHot
        n = len(data)
        self.X[DataType] = np.zeros( (2, n))
        self.y[DataType] = np.zeros( (1, n))
        self.yOneHot[DataType] = np.zeros( (3, n))
        for iter in range(n):
            self.X[DataType][0:2, iter] = data[iter][0:2]
            self.y[DataType][0, iter] = data[iter][2]
            self.yOneHot[DataType][data[iter][2]-1, iter] = 1

        # Assign regressor to 0'th activation layer
        self.a['0'] = self.X[DataType]

    def InitializeParameters(self, InitializeMethod):
        '''Initialize the bias vectors and weight matrices with InitializeMethod being default/normalized'''

        for l in range(1, self.NumberOfLayers+1):
            # Initialize b
            self.b[str(l)] = np.zeros( (self.NeuronsList[str(l)],1) )
            # Initialize W
            if InitializeMethod=='default':
                bound = np.sqrt(self.NeuronsList[str(l-1)])
            elif InitializeMethod=='normalized':
                bound = np.sqrt( 6/(self.NeuronsList[str(l-1)]+self.NeuronsList[str(l)]) )
            else:
                raise ValueError('Neural network parameter Initialization should be default or normalized.')
            self.W[str(l)] = np.random.uniform(-bound, bound, size = (self.NeuronsList[str(l)],self.NeuronsList[str(l-1)]))

    def Prop_Forward(self, DataType):
        '''Forward propagation of neural network'''

        # Dimensions
        n = self.yOneHot[DataType].shape[1]

        # Single forward propagation step
        for l in range(1, self.NumberOfLayers+1):
            print(l)
            # Linear transform
            self.z[str(l)] = self.W[str(l)]@self.a[str(l-1)]+self.b[str(l)]
            # Apply activation function
            self.a[str(l)] = self.fList[l]( self.z[str(l)] )

        # Compute cost
        self.cost = -np.sum(self.yOneHot[DataType]*np.log(self.a[str(self.NumberOfLayers)]), axis=None)/n




#------------#
#--- MAIN ---#
#------------#

# Initialize neural network instance
MyNN = NeuralNetwork({'0': 2, '1': 50, '2': 50, '3': 3})
print(MyNN)

# Read Data
MyNN.ReadHRInitialsData('train')
#MyNN.ReadHRInitialsData('valid')
#MyNN.ReadHRInitialsData('test')

# Initialize parameters
MyNN.InitializeParameters('normalized')

# Single forward propagation
MyNN.Prop_Forward('train')
print(MyNN.cost)


#--------------------------------------------#
# This is A
print(MyNN.a['3'])
print(MyNN.yOneHot['train'])

#print(MyNN.y['valid'])
#print(MyNN.yOneHot['train'])
