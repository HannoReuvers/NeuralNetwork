#!/usr/local/bin/python3
from NeuralNetwork import NNclass
import numpy as np
import unittest

class TestNeuralNetwork(unittest.TestCase):
    def test_gradient(self):
        '''Check backward propagation against numerical derivative'''

        # Small neural network for testing
        testNN = NNclass({'0': 2, '1': 10, '2': 5, '3': 3})
        testNN.ReadHRInitialsData('train')
        testNN.InitializeParameters('normalized')
        testNN.Prop_Forward('train')
        testNN.Prop_Backward('train')
        dWCalc, dbCalc = testNN.NumericalGradient('train')

        GradientOK = True
        for l in range(1,4):

            # Gradient comparison norms
            ErrorInb = np.linalg.norm( testNN.db[str(l)]-dbCalc[str(l)])
            ErrorInW = np.linalg.norm( testNN.dW[str(l)]-dWCalc[str(l)])
            if ErrorInb>1E-8 or ErrorInW>1E-8:
                GradientOK = False

            # Inform user
            print('\nLayer: ', l)
            print('Norm gradient difference in db: ', ErrorInb )
            print('Norm gradient difference in dW: ', ErrorInW )

        self.assertLessEqual(GradientOK, True)

unittest.main()
