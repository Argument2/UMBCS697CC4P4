import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
from sklearn import cross_validation
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where

def BuildDataset(X,y):
    ds = ClassificationDataSet(X.shape[1],1,nb_classes=4)
    
    for i in range(0, len(X)):
        ds.addSample(X[i],y[i])
        
    
    ds._convertToOneOfMany()    
    return ds
    
def CompareTargets(test,truth):
    correct=0
    for i in range(0,len(test)):
        if test[i]==truth[i]:
            correct += 1
    
    return float(correct) / float(len(test))

#----- Main Program -----

outputdir = '../Output/ANN/'
bcdatafile = '../Data/BreastCancerOriginal/breast_cancer.csv'
bctarget = '../Data/BreastCancerOriginal/breast_cancer_target_k4_100runs.csv'

X = np.loadtxt(open(bcdatafile, 'rb'), delimiter=',',skiprows=1)
bctarget = np.loadtxt(open(bctarget, 'rb'), delimiter=',')
y = bctarget.T

ds = BuildDataset(X,y)

traindata, testdata = ds.splitWithProportion(.1)

net = buildNetwork(len(X[0]), 500, 4, outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, traindata)
trainer.trainEpochs(5)
    
out = net.activateOnDataset(testdata)

trnresult = percentError( trainer.testOnClassData(),
                              traindata['class'] )
                              
tstresult = percentError( trainer.testOnClassData(
           dataset=testdata ), testdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult