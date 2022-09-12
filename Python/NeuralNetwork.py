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

# Gradient functions
gradsigmoid = lambda x: sigmoid(x)*(1- sigmoid(x))
gradTanh = lambda x: 1-Tanh(x)*Tanh(x)
gradReLu = lambda x: 1.0*(x>0)

#----------------------------#
#--- NEURAL NETWORK CLASS ---#
#----------------------------#
class NNclass:

    # Constants
    MaxEpochs = 20000

    # Activation function (None signifies to transformation to input)
    fList = [None, Tanh, Tanh, softmax]
    gradList = [None, gradTanh, gradTanh, None]

    # Initialize empty lists
    X = {}
    y = {}
    yOneHot = {}
    a = {}; da = {}
    b = {}; db = {}
    W = {}; dW = {}
    z = {}; dz = {}
    cost = {}

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

        #--- Single forward propagation step ---
        for l in range(1, self.NumberOfLayers+1):
            # Linear transform
            self.z[str(l)] = self.W[str(l)]@self.a[str(l-1)]+self.b[str(l)]
            # Apply activation function
            self.a[str(l)] = self.fList[l]( self.z[str(l)] )

        # Compute cost
        self.cost = -np.sum(self.yOneHot[DataType]*np.log(self.a[str(self.NumberOfLayers)]), axis=None)/n

        return self.cost

    def Prop_Backward(self, DataType):
        '''Backward propagation of neural network'''

        # Dimensions
        n = self.yOneHot[DataType].shape[1]
        L = self.NumberOfLayers

        #--- Single backward propagation step ---
        # Softmax layer
        self.dz[str(L)] = self.a[str(L)] - self.yOneHot[DataType]
        self.db[str(L)] = np.sum(self.dz[str(L)], axis = 1, keepdims = True)/n
        self.dW[str(L)] = ( self.dz[str(L)]@np.transpose(self.a[str(L-1)]) )/n

        # Fully connected layers 1,...,L-1
        for l in range(L-1,0,-1):
            self.da[str(l)] = np.transpose(self.W[str(l+1)])@self.dz[str(l+1)]
            self.dz[str(l)] = self.da[str(l)]*self.gradList[l]( self.z[str(l)] )
            self.db[str(l)] = np.sum(self.dz[str(l)], axis = 1, keepdims = True)/n
            self.dW[str(l)] = ( self.dz[str(l)]@np.transpose(self.a[str(l-1)]) )/n

    def NumericalGradient(self, DataType):
        '''Calculate numerical gradients for W and b for neural network'''

        # Increment for numerical difference
        EPSILON = 1E-5

        # Initialize output
        dWNum = {}
        dbNum = {}
        for l in range(1, self.NumberOfLayers+1):
            dWNum[str(l)] = np.zeros( np.shape(self.dW[str(l)]) )
            dbNum[str(l)] = np.zeros( np.shape(self.db[str(l)]) )

        # Calculate numerical gradients in b
        for l in range(1, self.NumberOfLayers+1):
            nRows = len( dbNum[str(l)] )
            for iter in range(0, nRows):

                # Increment b elements and compute cost function
                self.b[str(l)][iter] = self.b[str(l)][iter]+EPSILON         # Add EPSILON to get +EPSILON
                costPlus = self.Prop_Forward(DataType)
                self.b[str(l)][iter] = self.b[str(l)][iter]-2*EPSILON       # Subtract 2*EPSILON to get -EPSILON
                costMinus = self.Prop_Forward(DataType)

                # Numerical gradient
                dbNum[str(l)][iter] = (costPlus-costMinus)/(2*EPSILON)

                # Reset b
                self.b[str(l)][iter] = self.b[str(l)][iter] + EPSILON       # Add EPSILON to undo all operations

        # Calculate numerical gradients in W
        for l in range(1, self.NumberOfLayers+1):
            nRows, nCols = np.shape( self.dW[str(l)] )
            for rowiter in range(0, nRows):
                for coliter in range(0, nCols):

                    # Increment W elements and compute cost functions
                    self.W[str(l)][rowiter, coliter] = self.W[str(l)][rowiter, coliter]+EPSILON
                    costPlus = self.Prop_Forward(DataType)
                    self.W[str(l)][rowiter, coliter] = self.W[str(l)][rowiter, coliter]-2*EPSILON
                    costMinus = self.Prop_Forward(DataType)

                    # Numerical gradient
                    dWNum[str(l)][rowiter, coliter] = (costPlus-costMinus)/(2*EPSILON)

                    # Reset W
                    self.W[str(l)][rowiter, coliter] = self.W[str(l)][rowiter, coliter]+EPSILON

        # Return numerical gradients to user
        return dWNum, dbNum


#------------#
#--- MAIN ---#
#------------#
