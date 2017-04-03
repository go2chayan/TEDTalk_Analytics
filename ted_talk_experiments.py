import ted_talk_sentiment as ts
from list_of_talks import allrating_samples
import ted_talk_cluster_analysis as tca
import ted_talk_prediction as tp
from sklearn.cluster import KMeans, DBSCAN
import sklearn as sl
import scipy as sp
import matplotlib.pyplot as plt
import numpy

# This python file is for enlisting all the experiments we are doing
# It can also be used as sample usage of the code repository such as
# the sentiment_comparator class. Instead of running this file like a python
# script, try running it like a python notebook. For each experiment,
# try running the contents of each function in an interactive python shell.
# In that way, you don't need to re-execute time intensive functions.
###################################################################
# DO NOT delete an experiment even though they are highly redundant
###################################################################
# Bluemix sentiments:
# ==================
# 0: anger 
# 1: disgust 
# 2: fear 
# 3: joy 
# 4: sadness 
# 5: analytical 
# 6: confident 
# 7: tentative 
# 8: openness_big5 
# 9: conscientiousness_big5 
# 10: extraversion_big5 
# 11: agreeableness_big5 
# 12: emotional_range_big5


comparator = ts.Sentiment_Comparator(
    ts.hi_lo_files,     # Compare between hi/lo viewcount files
    ts.read_bluemix,    # Use bluemix sentiment
    )

# Check the average emotion plots
def bluemix_plot1():
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[0,1,2,3,4],   # only emotion scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center'
        )

# Check the average Language plots
def bluemix_plot2():
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[5,6,7],   # only Language scores
        styles=['r.--','r-','r--',
                'b.--','b-','b--'],  # appropriate line style
        legend_location='lower center'
        )

# Check the average social plots
def bluemix_plot3():
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[8,9,10,11,12],   # only big5 scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center'
        )

# bluemix plot one by one
def bluemix_plot4():
    avg_ = comparator.calc_group_mean()
    for i in range(13):
        # Plot Group Average
        ts.draw_group_mean_sentiments(avg_, # the average of groups
            comparator.column_names,        # name of the columns
            selected_columns=[i],   # only emotion scores
            styles=['r-',
                    'b-'],  # appropriate line style
            legend_location='lower center',
            outfilename='./plots/'+comparator.column_names[i]+'.pdf'
            )

# Checking time averages to see the best sentiment
def bluemix_plot5():
    avg_,p = comparator.calc_time_mean()
    ts.draw_time_mean_sentiments(avg_, # time averages
        comparator.column_names,       # name of the columns
        p,                              # p values
        outfilename='./plots/hivi_lovi.pdf'
        )

# Checking the Emotional progression for a single talk
def single_plot():
    talkid = 66
    selected_columns = [1,3,12]
    # Display sample sentences
    comparator.display_sentences(talkid,
        17, # Start percent
        29,  # End percent (also try 90 to 100)
        selected_columns     # Show only Disgust, Joy and Emotional
        )

    ts.draw_single_sentiment(\
        comparator.sentiments_interp[talkid], # plot the interpolated sentiment
        comparator.column_names,              # Name of the columns
        selected_columns                      # Show only Disgust, Joy and Emotional 
        )
    
# See the sentences of a talk from a certain percent to another percent
def see_sentences():
    # Display sample sentences
    comparator.display_sentences(66, # Talk ID
        50, # Start percent
        60  # End percent
        )

# Experiment on High/Low ratings
# Version edited by Samiha:
def time_avg_hi_lo_ratings():
    avg_saved = numpy.array([])
    i = 0
    for a_grp_dict in allrating_samples:
        i = i+1
        allkeys = sorted(a_grp_dict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]
        print titl
        compar = ts.Sentiment_Comparator(
            a_grp_dict,     # Compare between hi/lo viewcount files
            ts.read_bluemix,    # Use bluemix sentiment
            )
        avg_,p = compar.calc_time_mean()
        avg_saved = numpy.append(avg_saved, avg_)
        ###----------------had to close it to access radarplot.py------------
        #ts.draw_time_mean_sentiments(avg_, # time averages
         #   comparator.column_names,       # name of the columns
          #  p,                             # p values                      
           # outfilename='./plots/'+titl+'.pdf'
            #)
        ##----------------------------------------------------------------
 
    return avg_saved

# Experiment on High/Low ratings
def time_avg_hi_lo_ratings_original():
    avg_saved = numpy.array([])
    for a_grp_dict in allrating_samples:
        allkeys = sorted(a_grp_dict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]
        print titl
        compar = ts.Sentiment_Comparator(
            a_grp_dict,     # Compare between hi/lo viewcount files
            ts.read_bluemix,    # Use bluemix sentiment
            )
        avg_,p = compar.calc_time_mean()
        ts.draw_time_mean_sentiments(avg_, # time averages
           comparator.column_names,       # name of the columns
           p,                             # p values                      
           outfilename='./plots/'+titl+'.pdf'
            )

# Experiment on High/Low rating from group average
def grp_avg_hilo_ratings(score_list=[[0,1,2,3,4],[5,6,7],[8,9,10,11,12]]):
    for a_grpdict in allrating_samples:
        allkeys = sorted(a_grpdict.keys())
        titl = allkeys[0]+' vs. '+allkeys[1]+' group average'
        print titl
        compar = ts.Sentiment_Comparator(
            a_grpdict,     # Compare between hi/lo viewcount files
            ts.read_bluemix,    # Use bluemix sentiment
            )
        grp_avg = compar.calc_group_mean()
        for i in score_list:
            if len(i)==3:
                styles = ['r^-','r--','r-',
                 'b^-','b--','b-']
            else:
                styles = ['r^-','r--','r-','r.-','ro-',
                 'b^-','b--','b-','b.-','bo-']

            ts.draw_group_mean_sentiments(grp_avg,
                compar.column_names,
                i,
                styles,
                outfilename='./plots/'+titl+'.pdf')

# Experiment on the global average of sentiment progressions in
# ALL* tedtalks
# * = all means the 2007 valid ones.
def draw_global_means(X,comp):
    #X,comp = tca.load_all_scores()
    inp_dict = comp.groups.copy()
    inp_dict['all']=[]
    inp_dict['all'].extend(inp_dict['group_1'])
    inp_dict['all'].extend(inp_dict['group_2'])
    del inp_dict['group_1']
    del inp_dict['group_2']
    comp.reform_groups(inp_dict)
    avg = comp.calc_group_mean()
    ts.draw_group_means(avg,comp.column_names,\
        outfilename='./plots/global_mean.pdf')

# Experiment on kmeans clustering
# Practically all of them becomes flat line. Bad.
def kmeans_clustering(X,comp):
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = KMeans(n_clusters=5)
    clust_dict = tca.get_clust_dict(X,km,comp)    
    comp.reform_groups(clust_dict)
    avg = comp.calc_group_mean()
    ts.draw_group_means(avg,comp.column_names,\
        outfilename='./plots/cluster_mean.pdf')

# Experiment on kmeans clustering separately on each sentiment score
# check details on March 19th note in the TED Research document.
# It has a little re-computation which I just left alone.
def kclust_separate_stand(X,comp):
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps=6.5)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    avg_dict=tca.clust_separate_stand(X,km,comp,\
        csvcontent,csv_vid_idx)
    tca.draw_clusters(avg_dict,comp.column_names,
        outfilename='./plots/standardizedcluster_mean.pdf')    

# Draw the top 20 talks most similar to the cluster means
# and name five of them
def clusters_pretty_draw(X,comp):
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps=6.5)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    avg_dict=tca.clust_separate_stand(X,km,comp,\
        csvcontent,csv_vid_idx)
    tca.draw_clusters_pretty(avg_dict,comp,csvcontent,csv_vid_idx)    

# GOOD Result
##############
# Draw the cluster means and evaluate the differences in various
# clusters. It performs an ANOVA test to check if the clusters have
# any differences in their ratings
def evaluate_clusters_pretty(X,comp):
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps=7,min_samples=8)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    tca.evaluate_clust_separate_stand(X,km,comp,csvcontent,csv_vid_idx)
    

def classify_Good_Bad(scores,Y,classifier='LinearSVM'):
    #scores,Y = tp.loaddata()
    X,nkw = tp.feat_sumstat(scores)
    for i,kw in enumerate(tp.kwlist):
        print
        print
        print kw
        print '================='
        y = tp.discretizeY(Y,i)
        X_bin,y_bin = tp.binarize(X,y)
        # Split in training and test data
        tridx,tstidx = tp.traintest_idx(len(y_bin))
        trainX,trainY = X_bin[tridx,:],y_bin[tridx]
        testX,testY = X_bin[tstidx,:],y_bin[tstidx]

        # Classifier selection
        if classifier == 'LinearSVM':
            clf = sl.svm.LinearSVC()
            # Train with training data
            try:
                clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=10.)},nb_iter=10,datname=kw)
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC on Test Data'+kw)
        elif classifier == 'SVM_rbf':
            clf = sl.svm.SVC()
            # Train with training data
            clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                {'C':sp.stats.expon(scale=10.),},nb_iter=10,datname=kw)




if __name__=='__main__':
    # bluemix_plot1()
    # bluemix_plot2()
    # bluemix_plot3()
    # bluemix_plot4()
    # bluemix_plot5()
    # single_plot()
    # time_avg_hi_lo_ratings_original()
    # grp_avg_hilo_ratings([[1,2],[5,6],[10,12]])
    # -------- Clustering Experiments ---------
    # X,comp = tca.load_all_scores()
    # draw_global_means(X,comp)
    # kmeans_clustering(X,comp)
    # kmeans_separate_stand(X,comp)
    # evaluate_clusters_pretty(X,comp)
    # -------- Classification Experiments -----
    scores,Y = tp.loaddata()
    classify_Good_Bad(scores,Y)
