import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.utilities import percentError
from sklearn import cross_validation
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where

def BuildDataset(X,y):
    ds = ClassificationDataSet(X.shape[1],nb_classes=4)
    
    for i in range(0, len(X)):
        ds.addSample(X[i],y[i])
    
    ds._convertToOneOfMany()    
    return ds
    
def CompareTargets(test,truth):
    correct=0
    for i in range(0,len(test)):
        if test[i].argmax()==truth[i].argmax():
            correct += 1
    
    return float(correct) / float(len(test))

def BuildClusteringMatrixProb(predictions, n_clusters):
    cm = np.zeros(shape = (len(predictions),n_clusters))
    counter = 0
    
    for p in predictions:
        cm[counter,np.argmax(p)]=1
        counter += 1
        
    return cm
    
def GetOverallAccuracy(accuracies):
    total = 0.0
    for key in accuracies:
        total += float(accuracies[key])
        
    return total / float(len(accuracies))
    
    

#----- Main Program -----

outputdir = '../Output/ANN/'
bcdatafile = '../Data/BreastCancerOriginal/breast_cancer.csv'
bctarget = '../Data/BreastCancerOriginal/breast_cancer_target_k4_100runs.csv'

X = np.loadtxt(open(bcdatafile, 'rb'), delimiter=',',skiprows=1)
bctarget = np.loadtxt(open(bctarget, 'rb'), delimiter=',')
y = bctarget.T

totalsamples = len(X)

ds = BuildDataset(X,y)

accuracies = {}

for i in range(0,10):
    traindata, testdata = ds.splitWithProportion(0.9)
    net = buildNetwork(traindata.indim, 500, traindata.outdim, outclass=SoftmaxLayer)
    
    trainer = BackpropTrainer(net, dataset=traindata, momentum=0.1, verbose=True, weightdecay=0.01)
    trainer.trainEpochs(5)

    out = net.activateOnDataset(testdata)
    testtarget = testdata['target']

    accuracy = CompareTargets(BuildClusteringMatrixProb(out,4),testtarget)
    accuracies[i]=accuracy
    
print accuracies
print GetOverallAccuracy(accuracies)

