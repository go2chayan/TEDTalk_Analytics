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

def load_all_scores():
    '''
    This function loads all the valid TED talks in two groups. The
    groups are arbitrarily formed by just splitting the list in two halves.
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

def get_clust_dict(X,clusterer,comparator):
    '''
    Similar to get_clust_labels, but instead of returning the cluster
    labels, it returns a dictionary (which can be fed into the sentiment 
    comparator class). All the talksids are regrouped according to the
    cluster labels.
    '''
    result_dict = {}
    labls = get_clust_labels(X,clusterer)
    for lab,talkid in zip(labls,comparator.alltalks):
        if result_dict.get('cluster_'+str(lab)):
            result_dict['cluster_'+str(lab)].append(talkid)
        else:
            result_dict['cluster_'+str(lab)]=[talkid]
    return result_dict