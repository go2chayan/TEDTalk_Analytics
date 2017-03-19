'''
The goal of this module is to divide the ted talks in a number of
"clusters" so that the mean of each cluster does not become a flat
line. Instead, they should indicate some (storytelling) pattern.

We'll try various clustering options, starting from the simplest 
ones. A particularly interesting one is agglomerative clustering
where we can specify connectivity constraints. LPP is also similar.
Another interesting clustering approach would be to add constraints
to push the mean away from the flat-zero. This is interesting in
order to avoid the zero-mean phenomenon described in the following
paper:
http://www.cs.ucr.edu/~eamonn/meaningless.pdf

Silhouette Coef. will be used to tune the model parameters. However,
the ultimate evaluation metric is obviously the patterns of group mean.
'''

from list_of_talks import all_valid_talks
from ted_talk_sentiment import Sentiment_Comparator, read_bluemix
import numpy as np
import matplotlib.pyplot as plt

def load_score_array():
    '''
    The score array has a shape N x M x B, where N is the total
    number of talks (2007), M is the interpolated length of each talk (100)
    and B is the number of Bluemix Scores (13).
    Note: This function takes time
    '''
    # Let's form an input to sentiment comparator
    m = len(all_valid_talks)
    dict_input = {'group_1':all_valid_talks[:m/2],
                  'group_2':all_valid_talks[m/2:]}
    # Load into sentiment comparator for all the pre-comps
    comp = Sentiment_Comparator(dict_input,read_bluemix)
    X = np.array([comp.sentiments_interp[atalk] for atalk in comp.alltalks])
    return X,comp

def get_clust_labels(X,clusterer):
    '''
    Return the labels of clusters for various datapoints
    '''
    N,M,B = X.shape
    Z = X.reshape((N,M*B))
    clusterer.fit(Z)
    return clusterer.labels_

def get_clust_dict(labls,comparator):
    '''
    Returns a dictionary (as the input of sentiment comparator demands)
    from the cluster labels. All the talksids are grouped by cluster labels.
    '''
    result_dict = {}
    for lab,talkid in zip(labls,comparator.alltalks):
        if result_dict.get('cluster_'+str(lab)):
            result_dict['cluster_'+str(lab)].append(talkid)
        else:
            result_dict['cluster_'+str(lab)]=[talkid]
    return result_dict

def draw_cluster_avg(X,comp,clusterer):
    '''
    Draw the averages of various clusters. It creates as many figures
    as many seniment scores are there (13 for Bluemix).
    '''
    labels = get_clust_labels(X,clusterer)
    clust_dict = get_clust_dict(labels,comp)    
    comp.reform_groups(clust_dict)
    avg = comp.calc_group_mean()
    m,n = avg[avg.keys()[0]].shape
    # Draw the cluster averages
    for acol in range(n):
        plt.figure(figsize=(16,9))
        for akey in avg:
            plt.plot(avg[akey][:,acol],label=akey)
        plt.xlabel('Percent of Talk Progression')
        plt.ylabel('Value')
        plt.title(comp.column_names[acol])
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, right=0.99, left=0.05, top=0.85)
        plt.legend(bbox_to_anchor=(0., 1.05, 1., 0), loc=3,\
           ncol=5, mode="expand", borderaxespad=0.)
    plt.show()

