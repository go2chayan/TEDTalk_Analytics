import ted_talk_sentiment as ts
from list_of_talks import allrating_samples,all_valid_talks
import ted_talk_cluster_analysis as tca
import ted_talk_prediction as tp
from ted_talk_statistic import plot_statistics
from ted_talk_statistic_correlation import plot_correlation
from sklearn.cluster import KMeans, DBSCAN
import sklearn as sl
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

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

def bluemix_plot1(outfilename = None):
    '''
    This function plots the progression of average <b>emotion scores</b>
    for 30 highest viewed ted talks and 30 lowest viewed ted talks.
    If you want to save the plots in a file, set the outfilename argument.
    '''
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[0,1,2,3,4],   # only emotion scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )

def bluemix_plot2(outfilename=None):
    '''
    This function plots the progression of average Language scores for 30 
    highest viewed ted talks and 30 lowest viewed ted talks. If you want
    to save the plots in a file, set the outfilename argument.
    '''
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[5,6,7],   # only Language scores
        styles=['r.--','r-','r--',
                'b.--','b-','b--'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )

def bluemix_plot3(outfilename=None):
    '''
    This function plots the progression of average Social scores for 30 
    highest viewed ted talks and 30 lowest viewed ted talks. If you want
    to save the plots in a file, set the outfilename argument.
    '''
    avg_ = comparator.calc_group_mean()
    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[8,9,10,11,12],   # only big5 scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center',
        outfilename=outfilename
        )


def bluemix_plot4(outprefix='./plots/'):
    '''
    This function plots the progression of all the scores one by one.
    The average was calculated for 30 highest viewed ted talks and 30
    lowest viewed ted talks. By default, the plots are saved with their
    unique names inside the directory specified by outprefix argument.
    If you want to see the plots in window, set outprefix to None
    '''
    avg_ = comparator.calc_group_mean()
    for i in range(13):
        if outprefix:
            outfname = './plots/'+comparator.column_names[i]+'.pdf'
        else:
            outfname = None
        # Plot Group Average
        ts.draw_group_mean_sentiments(avg_, # the average of groups
            comparator.column_names,        # name of the columns
            selected_columns=[i],   # only emotion scores
            styles=['r-',
                    'b-'],  # appropriate line style
            legend_location='lower center',
            outfilename=outfname)

def bluemix_plot5(outfilename='./plots/hivi_lovi.pdf'):
    '''
    This function plots the time averages for the 30 highest viewed
    and 30 lowest viewed ted talks. In addition, it performs T-tests
    among the hi-view and lo-view groups. By default, the output is saved
    in the './plots/hivi_lovi.pdf' file. But if you want to see it
    on an interactive window, just set outfilename=None
    '''
    avg_,p = comparator.calc_time_mean()
    ts.draw_time_mean_sentiments(avg_, # time averages
        comparator.column_names,       # name of the columns
        p,                              # p values
        outfilename=outfilename
        )

def single_plot(talkid = 66,selected_scores = [1,3,12],
    draw_full_y=False,outfilename=None):
    '''
    Plots the score progression for a single talk.
    Note that this function does not plot the raw score.
    It smoothens the raw score value, cuts the boundary distortions
    (due to smoothing) and interpolates from 0 to 100 before showing
    the plots.
    The selected_scores argument defines which scores to show. Showing
    too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the selected
    scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk,ts.read_bluemix)
    ts.draw_single_sentiment(\
        comp.sentiments_interp[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )

def single_plot_raw(talkid,selected_scores=[3,4],
    draw_full_y=False,outfilename=None):
    '''
    Plots the <b>Raw</b> score progression for a single talk.
    The selected_scores argument defines which scores to show. Showing
    too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the 
    selected_scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk,ts.read_bluemix,process=False)
    comp.extract_raw_sentiment()
    ts.draw_single_sentiment(\
        comp.raw_sentiments[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )

def single_plot_smoothed(talkid,selected_scores=[3,4],
    draw_full_y=False,outfilename=None):
    '''
    Plots the Smoothed (but not interpolated) score progression for a
    single talk. The selected_scores argument defines which scores to 
    show. Showing too many scores at once will make the plot busy.
    If draw_full_y is set True, the plots are drawn over a y-axis ranging
    from 0 to 1.
    If outfilename is set to a filename, the plot is saved to that file.
    The indices of bluemix scores are as follows (needed in the 
    selected_scores argument):
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5
    '''
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk,ts.read_bluemix)
    ts.draw_single_sentiment(\
        comp.raw_sentiments[talkid], # plot the interpolated sentiment
        comp.column_names,              # Name of the columns
        selected_scores,                # Show only Disgust, Joy and Emotional
        full_y=draw_full_y,
        outfilename = outfilename 
        )    
    
def see_sentences_percent(talkid,start=50,end=60,selected_scores=None):
    '''
    Prints the sentences of a talk from a start percent to end percent.
    Notice that the start and end indices are numbered in terms of
    percentages of the the talk. The percentages are automatically
    converted back to the raw indices of each sentence.
    This function also shows the scores for each sentence. Use the
    selected_scores argument to specify which scores you want to see.
    By default, it is set to None, which means to show all the scores
    for each sentence.
    '''
    # Display sample sentences
    singletalk = {'just_one':[talkid]}
    comp = ts.Sentiment_Comparator(singletalk,ts.read_bluemix)
    comp.display_sentences(talkid, # Talk ID
        start, # Start percent
        end,  # End percent
        selected_columns = selected_scores
        )

def time_avg_hi_lo_ratings():
    '''
    Experiment on High/Low ratings
    This Version was edited by Samiha
    '''
    avg_saved = np.array([])
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
        avg_saved = np.append(avg_saved, avg_)
        ###----------------had to close it to access radarplot.py------------
        #ts.draw_time_mean_sentiments(avg_, # time averages
         #   comparator.column_names,       # name of the columns
          #  p,                             # p values                      
           # outfilename='./plots/'+titl+'.pdf'
            #)
        ##----------------------------------------------------------------
 
    return avg_saved

def time_avg_hi_lo_ratings_original():
    '''
    Experiment on the time average of (30) Highly rated talks and 
    low rated talks. 
    Besides calculating the time average, it also calculates
    the p-values for t-tests showing if there is any difference in 
    the average scores.
    The plots are saved in ./plots/ directory.
    '''
    avg_saved = np.array([])
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

def grp_avg_hilo_ratings(score_list=[[0,1,2,3,4],[5,6,7],[8,9,10,11,12]]):
    '''
    Experiment on the (ensemble) average of scores for 30 Highly rated
    talks and 30 low rated talks. 
    For every rating, it attempts to show the averages of various scores.
    The score_list is a list of list indicating which scores would be
    grouped together in one window. By default, the emotional, language,
    and personality scores are grouped together. The indices of the scores
    are given below:
    0: anger 
    1: disgust 
    2: fear 
    3: joy 
    4: sadness 
    5: analytical 
    6: confident 
    7: tentative 
    8: openness_big5 
    9: conscientiousness_big5 
    10: extraversion_big5 
    11: agreeableness_big5 
    12: emotional_range_big5

    The plots are saved in ./plots/ directory.
    '''
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
            if len(i)==1:
                styles = ['r-','b-']
            elif len(i)==2:
                styles = ['r^-','r--',
                 'b^-','b--']                
            elif len(i)==3:
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

def draw_global_means(comp):
    '''
    Experiment on the global average of sentiment progressions in
    ALL* tedtalks
    * = all means the 2007 valid ones.
    Use the following commands to generate comp where ts is the
    ted_talk_sentiment.py module
    comp = ts.Sentiment_Comparator({'all':all_valid_talks},ts.read_bluemix)
    '''
    avg = comp.calc_group_mean()['all']
    plt.figure(figsize=(6.5,6))
    grpnames = ['Emotion Scores', 'Language Scores', 'Personality Scores']
    for g,agroup in enumerate([[0,1,2,3,4],[5,6,7],[8,9,10,11,12]]):
        groupvals = np.array([avg[:,acol] for acol in agroup]).T
        import re
        colnames = [re.sub(\
            'emotion_tone_|language_tone_|social_tone_|_big5',\
            '',comp.column_names[acol]) for acol in agroup]

        plt.subplot(3,1,g+1)
        plt.plot(groupvals)
        plt.xlabel('Percent of Talk')
        plt.ylabel('Value')
        plt.ylim([[0,0.6],[0,0.5],[0.2,0.6]][g])
        #plt.subplots_adjust(bottom=0.05, right=0.99, left=0.05, top=0.85)
        #plt.legend(colnames,bbox_to_anchor=(0., 1.05, 1., 0), loc=3,\
        #   ncol=2, mode="expand", borderaxespad=0.)
        plt.legend(colnames,ncol=[5,3,3][g],loc=['upper left',\
            'upper left','lower left'][g])
        plt.title(['Emotion Scores','Language Scores','Personality Scores'][g])
        plt.tight_layout()
    plt.savefig('./plots/global_scores.pdf')

def kmeans_clustering(X,comp):
    '''
    Experiment on kmeans clustering

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module    
    load_all_scores is a slow function
    '''
    # Try Using any other clustering from sklearn.cluster
    km = KMeans(n_clusters=5)
    clust_dict = tca.get_clust_dict(X,km,comp)    
    comp.reform_groups(clust_dict)
    avg = comp.calc_group_mean()
    ts.draw_group_means(avg,comp.column_names,\
        outfilename='./plots/cluster_mean.pdf')

def kclust_separate_stand(X,comp):
    '''
    Experiment on kmeans clustering separately on each sentiment score.
    Check details on March 19th note in the TED Research document.
    It has a little re-computation which I just left alone.    

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module    
    load_all_scores is a slow function
    '''
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(pdf=6.25)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    avg_dict=tca.clust_separate_stand(X,km,comp,\
        csvcontent,csv_vid_idx)
    tca.draw_clusters(avg_dict,comp.column_names,
        outfilename='./plots/standardizedcluster_mean.pdf')    

def clusters_pretty_draw(X,comp):
    '''
    Draws the top 20 talks most similar to the cluster means
    and name five of them

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module
    load_all_scores is a slow function      
    '''
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(pdf=6.5)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    avg_dict=tca.clust_separate_stand(X,km,comp,\
        csvcontent,csv_vid_idx)
    tca.draw_clusters_pretty(avg_dict,comp,csvcontent,csv_vid_idx)    

# GOOD Result
##############
def evaluate_clusters_pretty(X,comp,outfilename='./plots/'):
    '''
    Draw the cluster means and evaluate the differences in various
    clusters. It performs an ANOVA test to check if the clusters have
    any differences in their ratings

    Note: before you call this function, you should get the arguments
    (X and comp) using the following command: 
    X,comp = tca.load_all_scores()
    tca is the ted_talk_cluster_analysis module
    load_all_scores is a slow function     
    '''
    #X,comp = tca.load_all_scores()
    # Try Using any other clustering from sklearn.cluster
    km = DBSCAN(eps=6.5)
    csvcontent,csv_vid_idx = tca.read_index(indexfile = './index.csv')
    tca.evaluate_clust_separate_stand(X,km,comp,csvcontent,
        csv_vid_idx,outfilename=outfilename)
    

def classify_Good_Bad(scores,Y,classifier='LinearSVM'):
    '''
    Classify between groups of High ratings and low ratings using
    Two different types of SVM LinearSVM or SVM_rbf. The classifier
    argument can take these two values.
    This function trains the classifiers and evaluates their performances.

    Use the following command to get the initial arguments:
    scores,Y,_ = tp.loaddata()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''
    X,nkw = tp.feat_sumstat(scores)
    for i,kw in enumerate(tp.kwlist):
        print
        print
        print kw
        print '================='
        print 'Predictor:',classifier
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
                    {'C':sp.stats.expon(scale=1.)},nb_iter=10,
                    datname = kw+'_LibSVM')
            except ImportError:
                raise
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'            
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of LinearSVM on Test Data for '+kw)
        elif classifier == 'SVM_rbf':
            clf = sl.svm.SVC()
            # Train with training data
            try:
                clf_trained,auc=tp.train_with_CV(trainX,trainY,clf,
                    {'C':sp.stats.expon(scale=25),
                    'gamma':sp.stats.expon(scale=0.05)},
                    nb_iter=100,datname=kw)
                print 'Number of SV:',clf_trained.n_support_
            except ImportError:
                raise
            except:
                print 'Data is badly scaled for',kw
                print 'skiping'
                continue
            # Evaluate with test data
            print 'Report on Test Data'
            print '-----------------------'                 
            # Evaluate with test data
            tp.classifier_eval(clf_trained,testX,testY,ROCTitle=\
                'ROC of SVM_RBF on Test Data for '+kw)

def classify_good_bad_raw_score():
    pass

def classify_multiclass():
    pass

def regress_totalviews_powerlaw():
    pass

def classify_totalviews_powerlaw():
    pass

def regress_ratings(scores,Y,regressor='ridge',cv_score=sl.metrics.r2_score):
    '''
    Try to predict the ratings using regression methods. Besides training
    the regressors, it also evaluates them.

    Use the following command to get the initial arguments:
    scores,Y,_ = tp.loaddata()
    tp = ted_talk_prediction module
    Note: loaddata is a slow function
    '''    
    X,nkw = tp.feat_sumstat(scores)
    for i,kw in enumerate(tp.kwlist):
        print
        print
        print kw
        print '================='
        print 'Predictor:',regressor
        y = Y[:,i]
        if kw == 'Totalviews':
            y=np.log(y)
        tridx,tstidx = tp.traintest_idx(len(y))
        trainX,trainY = X[tridx,:],y[tridx]
        testX,testY = X[tstidx,:],y[tstidx]

        # Predictor Selection
        if regressor=='ridge':
            # Train on training data
            rgrs = sl.linear_model.Ridge()
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'alpha':sp.stats.expon(scale=1.)},
                score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)
        elif regressor == 'SVR':
            # Train on training data
            rgrs = sl.svm.LinearSVR(loss='squared_pdfilon_insensitive',
                dual=False,pdfilon=0.001)
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'C':sp.stats.expon(scale=10)},
                score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)
        elif regressor == 'gp':
            # Train on training data
            rgrs = sl.gaussian_process.GaussianProcessRegressor()
            rgrs.fit(trainX,trainY)
            # Evaluate with test data
            print 'Report on Training Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs,testX,testY)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs,testX,testY)
        elif regressor == 'lasso':
            # Train on training data
            rgrs = sl.linear_model.Lasso()
            # Evaluate with test data
            print 'Report on Training Data:'
            print '-----------------------'             
            # Evaluate with training data
            rgrs_trained,score = tp.train_with_CV(trainX,trainY,
                rgrs,{'alpha':sp.stats.expon(scale=0.1)},score_func=cv_score)
            # Evaluate with test data
            print 'Report on Test Data:'
            print '-----------------------'             
            tp.regressor_eval(rgrs_trained,testX,testY)

if __name__=='__main__':
    # plot_statistics()
    # plot_correlation()
    # bluemix_plot1()
    # bluemix_plot2()
    # bluemix_plot3()
    # bluemix_plot4()
    # bluemix_plot5()
    # single_plot()
    # time_avg_hi_lo_ratings_original()
    # grp_avg_hilo_ratings([[1,2],[5,6],[10,12]])
    # -------- Clustering Experiments ---------
    X,comp = tca.load_all_scores()
    # draw_global_means(X,comp)
    # kmeans_clustering(X,comp)
    # kmeans_separate_stand(X,comp)
    evaluate_clusters_pretty(X,comp,outfilename='./plots/')
    # -------- Classification Experiments -----
    # scores,Y,_ = tp.loaddata()
    # classify_Good_Bad(scores,Y)
