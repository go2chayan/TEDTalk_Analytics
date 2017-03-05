import re
import os
import cPickle as cp
import numpy as np
from itertools import product
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dict of talks with highest view count and Lowest view counts
# Note, while calculating the lowest view counts, I took only
# the talks that are at least two years old (i.e. retention time
# is greater than 730 days). This is done to ignore the very new
# talks
hi_lo_files = {'High_View_Talks':[66,96,97,206,229,549,618,
685,741,848,1246,1344,1377,1569,1647,1815,1821,
2034,2399,2405],
'Low_View_Talks':[220,268,339,345,379,402,403,427,
439,500,673,675,679,925,962,1294,1332,1373,
1445,1466]}

############################ Generic Readers ##############################
# Read talk transcripts and tokenize the sentences
# While tokenizing, it removes sound tags: (Applause), (Laughter) etc.
def read_sentences(pklfile):
    assert os.path.isfile(pklfile),'File not found: '+pklfile
    data = cp.load(open(pklfile))
    txt = re.sub('\([a-zA-Z]*?\)','',' '.join(data['talk_transcript']))
    return sent_tokenize(txt)

# Similar to read_sentences, but returns utterances instead
def read_utterances(pklfile):
    assert os.path.isfile(pklfile),'File not found: '+pklfile
    data = cp.load(open(pklfile))
    txt = re.sub('\([a-zA-Z]*?\)','','__||__'.join(data['talk_transcript']))
    return txt.split('__||__')
######################## End of Generic Readers ###########################

################################ Generic Plotters ##########################
# Draws the sentiment values of a single talk. The input array
# can be either raw sentiments or interpolated sentiments
def draw_single_sentiment(anarray,outfilename=None):
    plt.figure()
    plt.plot(anarray)
    plt.tight_layout()
    plt.xlabel('Sentence Number')
    plt.ylabel('Values')
    plt.legend(['Negative','Neutral','Positive'])
    if outfilename:
        plt.savefig(outfilename)
    else:
        plt.show()

# Draws the ensemble averages of the sentiments
def draw_group_mean_sentiments(grp_means,
                          colnames=['neg','neu','pos'],
                          styles=['bo-','b--','b-','ro-','r--','r-'],
                          outfilename=None):
    plt.figure()
    for g,agroup in enumerate(grp_means):
        m,n = np.shape(grp_means[agroup])
        for col in range(n):
            plt.plot(grp_means[agroup][:,col],
                styles[g*len(colnames)+col],
                label=agroup+'_'+colnames[col])
        plt.xlabel('Interpolated Sentence Number')
        plt.ylabel('Values')
    plt.tight_layout()
    plt.legend(loc='center right')
    if outfilename:
        plt.savefig(outfilename)
    else:
        plt.show()

# Draw bar plots for time averages and annotate pvalues
def draw_time_mean_sentiments(time_avg,
                            pvals,
                            colnames=['neg','neu','pos'],
                            groupcolor=['royalblue','darkkhaki'],
                            outfilename=None):
    plt.figure()
    for i,grp in enumerate(time_avg):
        plt.bar(np.arange(len(time_avg[grp]))-i*0.25,
                time_avg[grp],
                color=groupcolor[i],
                width = 0.25,
                label=grp)
    plt.tight_layout()
    plt.ylabel('Average Sentiment Value')
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(np.arange(len(time_avg[grp])))
    ax.set_xticklabels(
        [c+': p='+str(p) for c,p in zip(colnames,pvals)])
    if outfilename:
        plt.savefig(outfilename)
    else:
        plt.show()
############################################################################

'''
Sentiment Comparator class contains all the information to compare
sentiment data between several groups of files. 
It retains the back references to the sentences. It contains the
following inputs and variables:
groups        : Input dictionary with group name as keys and the talk indices
                as values
reader        : reader takes a function. It can either take the read_sentences
                or the read_utterances -- indicating that the transcriptions
                would be read in sentence level or utterances level
inputFolder   : Folder where the .pkl files reside
raw_sentiments: A dictionary storing the raw sentiments (neg,neu,pos,uni)
                of each talk. The keys are the talk ids and values
                are matrices for which columns represent negative,
                neutral, positive and unified sentiments respectively.
                Unified sentiments are calculated by the following formula:
                Positive rating - Neutral rating
sentiments_intep: The sentiment data interpolated to a canonical size
back_ref      : Reference to the old indices of the sentences from 
                each interpolated sample
alltalks      : List of all the talk ids being analyzed in this comparator
'''
class Sentiment_Comparator(object):
    def __init__(self,dict_groups,reader,inputFolder='./talks/',process=True):
        self.inputpath=inputFolder
        self.reader = reader
        self.groups = dict_groups
        self.alltalks = [ids for agroup in self.groups \
            for ids in self.groups[agroup]]
        self.raw_sentiments = {}
        self.sentiments_intep={}
        self.back_ref={}
        if process:
            self.update_raw_sentiment()
            self.smoothen_raw_sentiment()
            self.update_sentiments_intep()

    # Fill out self.raw_sentiments
    def update_raw_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        for atalk in self.alltalks:
            sents = self.reader(self.inputpath+str(atalk)+'.pkl')
            values = []
            for asent in sents:
                results=analyzer.polarity_scores(asent)
                values.append([results['neg'],results['neu'],
                    results['pos']])
            self.raw_sentiments[atalk] = np.array(values)

    # Changes the self.raw_sentiments to a smoothed version
    def smoothen_raw_sentiment(self,kernelLen=5.):
        # Get number of columns in sentiment matrix 
        _,n = np.shape(self.raw_sentiments[self.alltalks[0]])
        for atalk in self.alltalks:
            for i in range(n):
                self.raw_sentiments[atalk][:,i] = np.convolve(\
                    self.raw_sentiments[atalk][:,i],\
                    np.ones(kernelLen)/kernelLen,mode='same')

    # Fill out the interpolated sentiments
    # It also updates the backward reference (back_ref)
    def update_sentiments_intep(self,bins=100):
        for atalk in self.alltalks:
            m,n = np.shape(self.raw_sentiments[atalk])
            # Pre-allocate
            self.sentiments_intep[atalk] = np.zeros((bins,n))
            # x values for the interpolation
            old_xvals = np.arange(m)
            new_xvals = np.linspace(0,old_xvals[-1],num=bins)
            # Update the backward reference
            self.back_ref[atalk] = [np.where((old_xvals>=lo) & \
                (old_xvals<=hi))[0].tolist() for lo,hi in \
                zip(new_xvals[:-1],new_xvals[1:])]+[[old_xvals[-1]]]
            # Interpolate column by column
            for i in range(n):
                self.sentiments_intep[atalk][:,i] = \
                np.interp(new_xvals,old_xvals,self.raw_sentiments[atalk][:,i])

    # Calculates (and returns) the ensemble averages of the groups
    def calc_group_mean(self):
        group_average = {}
        for agroup in self.groups:
            vals = [self.sentiments_intep[id] for id in self.groups[agroup]]
            # Averaging over the talks in a group
            group_average[agroup]=np.mean(vals,axis=0)
        return group_average

    # Calculates (and returns) the Time averages of the sentiments
    # Also returns the p-values if ttest is done. Note: ttest can't
    # be done for more than 2 groups
    def calc_time_mean(self,
                       perform_ttest=True,
                       column_names=['neg','neu','pos']):
        time_avg = {}
        for agroup in self.groups:
            vals = [self.sentiments_intep[id] for id in self.groups[agroup]]
            # Averaging over time
            time_avg[agroup]=np.mean(vals,axis=1)
        # Perform ttest for statistical significance
        if perform_ttest:
            pvals=[]
            m,n = np.shape(time_avg[agroup])
            allkeys=time_avg.keys()
            assert len(allkeys)==2,'T-test can not be done for 2+ groups'
            for i in range(n):
                print 'Sentiment:',column_names[i],
                _,p = ttest_ind(time_avg[allkeys[0]][:,i],
                    time_avg[allkeys[1]][:,i])
                print 'p-value:',p
                pvals.append(p)
        # Average and return
        for agroup in time_avg:
            time_avg[agroup]=np.mean(time_avg[agroup],axis=0)
        if not perform_ttest:
            return time_avg
        else:
            return time_avg,pvals


def main():
    #comparator = Sentiment_Comparator(hi_lo_files,read_utterances)
    comparator = Sentiment_Comparator(hi_lo_files,read_sentences)
    grp_avg = comparator.calc_group_mean()
    draw_group_mean_sentiments(grp_avg,outfilename='./plots/Ensemble_Avg_Sent.pdf')
    time_avg,pvals = comparator.calc_time_mean()
    draw_time_mean_sentiments(time_avg,pvals,outfilename='./plots/Time_Avg_Sent.pdf')
    



if __name__ == '__main__':
    main()