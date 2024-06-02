######################################
## Generate HRInitialsData as csv   ##
## Name: Hanno Reuvers              ##
## data: 2022/06/22                 ##
######################################
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# General parameters
create_csv = True
TypeOfData = 'valid' # select among 'train'/'valid'/'test'
# Set seed
if TypeOfData=='train':
    print('Creating training data')
    np.random.seed(2022) #<- Seed for training data
    Npoints = 5000
    print('Generating', Npoints, 'observations')
    name = 'HR_train.sqlite'
elif TypeOfData=='valid':
    print('Creating validation data')
    np.random.seed(1950) #<- Seed of development data
    Npoints = 1000
    print('Generating', Npoints, 'observations')
    name = 'HR_valid.sqlite'
elif TypeOfData=='test':
    print('Creating test data')
    np.random.seed(2050) #<- Seed of test data
    Npoints = 1000
    print('Generating', Npoints, 'observations')
    name = 'HR_test.sqlite'
Prob1 = 1
Prob2 = 1

# Encode H in matrix
Hprob = np.zeros( (10,10) )
Hprob[1:9, 1] = 1
Hprob[1:9, 4] = 1
Hprob[4:6, 2:4] = 1
Hprob = Prob1*Hprob

# Encode R in matrix
Rprob = np.zeros( (10,10) )
Rprob[1:9, 5] = 1
Rprob[1:5, 6:9] = 1
Rprob[5, 6] = 1
Rprob[6, 7] = 1
Rprob[7:9, 8] = 1
Rprob = Prob2*Rprob

# Remaining probabiliy
Backgroundprob = np.ones ( (10,10) ) - Hprob - Rprob

# Initialize output
X = np.empty( (2,Npoints) ); X[:] = np.NaN
Y = np.empty( (1,Npoints) ); Y[:] = np.NaN

# Generate data
for pointiter in range(Npoints):

    # Generate coordinates
    X[0, pointiter] = np.random.uniform()
    X[1, pointiter] = np.random.uniform()

    # Compute indices
    index1 = int(10-np.ceil( 10*X[1,pointiter] ))
    index2 = int(np.floor( 10*X[0,pointiter] ))

    # Assign label
    Value = np.random.uniform()
    if Value<=Hprob[index1,index2]:
        OutputClass = 1
    elif Value<=(Hprob[index1,index2]+Rprob[index1,index2]):
        OutputClass = 2
    else:
        OutputClass = 3

    Y[0, pointiter] = OutputClass


# Plot results
plt.scatter(X[0,:], X[1,:], c=Y)
plt.show()

# Use pandas library to store data as csv
if create_csv:
    df_data = pd.DataFrame()
    df_data["x"] = X[0,:]
    df_data["y"] = X[1,:]
    df_data["label"] = Y[0,:]
    df_data["label"] = df_data["label"].astype('int')
    df_data.to_csv("./"+TypeOfData+".csv", header=True, index=False)