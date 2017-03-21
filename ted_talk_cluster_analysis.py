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

def get_clust_dict(X,clusterer,comparator):
    '''
    Performs clustering and returns a dictionary (which can be fed into
    the sentiment comparator class). All the talksids are regrouped 
    according to the cluster labels.
    Note: It clusters with all the scores together. Don't use it.
    '''
    result_dict = {}
    N,M,B = X.shape
    Z = X.reshape((N,M*B))
    clusterer.fit(Z)    
    labls = clusterer.labels_
    for lab,talkid in zip(labls,comparator.alltalks):
        if result_dict.get('cluster_'+str(lab)):
            result_dict['cluster_'+str(lab)].append(talkid)
        else:
            result_dict['cluster_'+str(lab)]=[talkid]
    return result_dict

def clust_onescore_stand(X_1,clusterer,comparator):
    '''
    Similar to get_clust_dict. But it will performs clustering assuming there is
    only one sentiment score. Practically it is equivalent to considering that X_1
    is of order 2 (NxM), instead of 3 (NxMxB). In addition, it performs z-score 
    standardization of the rows of X_1 (i.e. each talk).
    '''
    result_dict = {}
    mean_ = np.mean(X_1,axis=1)[None].T
    std_ = np.std(X_1,axis=1)[None].T
    Z = (X_1-mean_)/std_
    clusterer.fit(Z)
    labls = clusterer.labels_
    for lab,talkid in zip(labls,comparator.alltalks):
        if result_dict.get('cluster_'+str(lab)):
            result_dict['cluster_'+str(lab)].append(talkid)
        else:
            result_dict['cluster_'+str(lab)]=[talkid]
    return result_dict

def clust_separate_stand(X,clusterer,comparator):
    '''
    It takes care of the scores individually. Although it is a bit slow
    due to some recomputations, but this would give better results in
    the clustering. Also, it z-score standardizes each TED talks signal
    which would reveal the storytelling patterns better.
    '''
    N,M,B = X.shape
    avg_dict = {}
    for s in range(B):
        clust_dict = clust_onescore_stand(X[:,:,s],clusterer,comparator)
        comparator.reform_groups(clust_dict)
        avg = comparator.calc_group_mean()
        # Although it computed the average for all the columns, I need
        # just one, s'th column. This is the recomputation. I don't
        # think it is too bad, though.
        for akey in avg:
            if not comparator.column_names[s] in avg_dict:
                avg_dict[comparator.column_names[s]] = {akey:avg[akey][:,s]}
            else:
                avg_dict[comparator.column_names[s]][akey]=avg[akey][:,s]
    return avg_dict
        

def draw_clusters(avg_dict,column_names,fullyaxis=False,\
        outfilename=None):
    '''
    This plotter expects the avg_dict from clust_separate_stand.
    '''
    for i,s in enumerate(avg_dict):
        plt.figure(figsize=(16,9))
        for akey in avg_dict[s]:
            plt.plot(avg_dict[s][akey],label=akey)
        plt.xlabel('Percent of Talk Progression')
        plt.ylabel('value')
        if fullyaxis:
            plt.ylim([0,1])
        plt.title(column_names[i])
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, right=0.99, left=0.05, top=0.85)
        plt.legend(bbox_to_anchor=(0., 1.05, 1., 0), loc=3,\
           ncol=5, mode="expand", borderaxespad=0.)
        if outfilename:
            import os
            split_fn = os.path.split(outfilename)
            plt.savefig(os.path.join(split_fn[0],column_names[i]+\
                '_'+split_fn[1]))
    if not outfilename:
        plt.show()

