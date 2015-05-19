""" @author edward.saad """
from __future__ import division
import csv,os,errno,pandas,operator
import numpy as np
import nimfa
import matplotlib.pyplot as plt
from time import time
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.pyplot import savefig, imshow, set_cmap


home_directory = os.path.abspath('..')
data_directory = home_directory + '/Data/Artificial/'
results_directory = home_directory + '/Output/Artificial/'
input_file = 'artificial50pct.csv'
header_file = ''

def readAndRun(input_file, header_file, output_directory, k_bottom, k_top, runs, algorithm):
    """
        reads the imput file and runs NMF.
    """
    print 'input file: ', input_file
    if algorithm == 'S':
        tumor = input_file.split('/')[-1].split('.')[0] + '_sparse'
    else:
        tumor = input_file.split('/')[-1].split('.')[0] + '_standard'
    print 'tumor name: ', tumor
    print 'output directory: ', output_directory
    try:
        os.makedirs(results_directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # read the input file
    V = np.loadtxt(open(input_file, 'rb'), delimiter=',', skiprows=1)
    # run NMF for all k's in [k_bottom, k_top]
    k_to_c = {}
    k_to_d = {}
    for rank in range(k_bottom, k_top):
        if algorithm == 'S':
            nmf_fit = runSNMF(V, rank, runs)
            consensus = nmf_fit.fit.consensus()
            print 'consensus: ', consensus
            cophenetic = nmf_fit.fit.coph_cor()
            dispersion = nmf_fit.fit.dispersion()
            print 'cophenetic correlation coefficient: ', cophenetic
            print 'dispersion coefficient: ', dispersion
            k_to_c[rank] = cophenetic
            k_to_d[rank] = dispersion
            p_consensus = reorder(consensus)
            plot(tumor, p_consensus, rank)
        else:
            nmf_fit = runNMF(V, rank, runs)
            consensus = nmf_fit.fit.consensus()
            print 'consensus: ', consensus
            cophenetic = nmf_fit.fit.coph_cor()
            dispersion = nmf_fit.fit.dispersion()
            print 'cophenetic correlation coefficient: ', cophenetic
            print 'dispersion coefficient: ', dispersion
            k_to_c[rank] = cophenetic
            k_to_d[rank] = dispersion
            p_consensus = reorder(consensus)
            plot(tumor, p_consensus, rank)
    optimal_rank = evaluateStability(k_to_c)
    plotCoefficientvRank(tumor, k_to_c, 'cophenetic')
    plotCoefficientvRank(tumor, k_to_d, 'dispersion')

def evaluateStability(rank_to_cophenetic):
    rank = max(rank_to_cophenetic.iteritems(), key=operator.itemgetter(1))[0]
    print 'the k with the highest cophenetic: ', rank   
    return rank

def runNMF(V, rank, runs):
    print 'rank: ', rank, 'runs: ', runs
    print 'matrix: ', V
    matrix = np.matrix(V).astype('float').T
    consensus = np.zeros((matrix.shape[1], matrix.shape[1]))
    nmf = nimfa.Nmf(matrix, rank=rank, seed="random_vcol", max_iter=200, update='euclidean',
                    objective='conn', conn_change=40, n_run=runs, track_factor=True)
    nmf_fit = nmf()
    return nmf_fit

def runSNMF(V, rank, runs):
    print 'rank: ', rank, 'runs: ', runs
    print 'matrix: ', V
    matrix = np.matrix(V).astype('float').T
    consensus = np.zeros((matrix.shape[1], matrix.shape[1]))
    nmf = nimfa.Snmf(matrix, rank=rank, seed="random_vcol", max_iter=200, update='euclidean',
                    objective='conn', conn_change=40, n_run=runs, track_factor=True)
    nmf_fit = nmf()
    return nmf_fit


def plot(tumor, consensus, rank):
    plt.figure("%s_consensus_rank_%d.png" % (tumor,rank))
    plt.subplot(211)
    plt.imshow(consensus)
    plt.set_cmap('RdBu_r')
    plt.savefig(results_directory + "%s_consensus_rank_%d.png" % (tumor,rank))


def plotCoefficientvRank(tumor, coefficient_to_rank, coefficient_type):
    print 'plotting coefficient vs rank for %s' % tumor
    print 'coefficient type: ', coefficient_type
    plt.figure("%s_" % tumor + "%s_v_rank.png" % coefficient_type)
    plt.subplot(211)
    plt.plot(coefficient_to_rank.keys(), coefficient_to_rank.values())
    plt.ylabel('coefficient ' + coefficient_type)
    plt.xlabel('Rank k')
    plt.savefig(results_directory + "%s_"% tumor + "%s_v_rank.png" % coefficient_type)


def reorder(C):
    print 'reorder...'
    Y = 1 - C
    Z = linkage(Y, method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]


def generateHeatPlot(tumor, header_list, matrix, W, H):
    plot_name = tumor + '_heat_plot'
    plt.figure(plot_name)
    # the matrix
    ax1 = plt.subplot(2,1,1)
    row_labels = header_list
    column_labels = ['' for i in range(len(matrix))]
    heatmap = ax1.pcolor(np.array(matrix), cmap=plt.cm.Blues)
    plt.gcf()
    ax1.set_xticklabels(column_labels, minor=False)
    ax1.set_yticklabels(row_labels, minor=False)
    ax2 = plt.subplot(2,2,3)
    ax2.pcolor(np.array(W), cmap=plt.cm.Blues)
    ax3 = plt.subplot(2,2,4)
    ax3.pcolor(np.array(H), cmap=plt.cm.Blues)
    print('Basis matrix:\n%s' % W)
    print('Mixture matrix:\n%s' % H)
    plt.savefig(results_directory + plot_name)


def main():
    # run it (input_file, output_directory, k_bottom, k_top, runs, algorithm)
    # run using Sparse NMF
    readAndRun(data_directory + input_file, data_directory + header_file, results_directory, 3, 7, 25, 'S')
    # run using Standard NMF
    #readAndRun(data_directory + input_file, data_directory + header_file, results_directory, 3, 7, 5, 'N')

if __name__ == '__main__':
    main()
