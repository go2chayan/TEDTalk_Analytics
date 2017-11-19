import cPickle as cp
import os
import re
import nltk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from list_of_talks import all_valid_talks
from TED_data_location import ted_data_path

def plot_statistics(infolder,outfolder):
    alltalks = [str(afile)+'.pkl' for afile in all_valid_talks]
    tottalks = len(alltalks)
    totlen,totratings,tottok,totsent = 0,0,0,0
    lenlst,viewlst,ratinglst,topratings,timealive,kwlst=[],[],{},{},[],[]
    for afile in alltalks:
        print afile
        atalk=cp.load(open(infolder+afile,'rb'))
        # Length of video in Minutes
        vidlength = float(atalk['talk_meta']['vidlen'])/60.
        lenlst.append(vidlength)
        totlen+=vidlength
        # Keyword list
        kwlst.append(atalk['talk_meta']['keywords'])
        # View count
        viewlst.append(atalk['talk_meta']['totalviews'])
        # Update total ratings and list the highest rating of each talk
        allratings = {key:val for key,val in atalk['talk_meta']['ratings'].items() \
            if not key=='total_count'}
        totalratcnt = atalk['talk_meta']['ratings']['total_count']
        totratings += totalratcnt
        for akey in allratings.keys():
            if not ratinglst.get(akey):
                ratinglst[akey] = [float(allratings[akey])/float(totalratcnt)*100.]
            else:
                ratinglst[akey].append(float(allratings[akey])/float(totalratcnt)*100.)
        toprat = max(allratings.items(),key=lambda x:x[1])[0]
        if not topratings.get(toprat):
            topratings[toprat]=1
        else:
            topratings[toprat]+=1
        # Total time the talk is alive before crawling
        dtdelta = atalk['talk_meta']['datecrawled'] - \
                atalk['talk_meta']['datepublished']
        timealive.append(dtdelta.days)
        # Total number of words and sentences
        fulltrns = ' '.join([aline.encode('ascii','ignore') for apara\
                in atalk['talk_transcript'] for aline in apara])
        fulltrns = re.sub('\([\w ]*?\)','',fulltrns)
        totsent+= len(nltk.sent_tokenize(fulltrns))
        tottok += len(nltk.word_tokenize(fulltrns))

    # Print total counts
    print 'Number of talks:',tottalks
    print 'Total length of all talks:',float(totlen)/60.,'Hours'
    print 'Total number of words:',tottok
    print 'Total number of ratings:',totratings
    print 'Total number of sentences:',totsent

    # Print averages
    print 'Average words per talk:', float(tottok)/float(tottalks)
    print 'Average sentences per talk:', float(totsent)/float(tottalks)
    print 'Average words per sentence:', float(tottok)/float(totsent)
    print 'Average length of talks:', float(totlen)/float(tottalks)/60.,'Minutes'

    # Plot video-length distribution
    plt.figure(1)
    plt.hist(lenlst,bins=50)
    plt.xlabel('Duration of the Talks (Minutes)')
    plt.ylabel('Number of Talks')
    plt.savefig(outfolder+'duration_hist.eps')

    # Plot viewcount distribution
    plt.figure(2)
    plt.hist(viewlst,bins=50)
    plt.xlabel('View Count')
    plt.ylabel('Number of Talks')
    plt.savefig(outfolder+'viewcount_hist.eps')

    # Plot number of talks with some specific ratings as maximum
    plt.figure(3)
    plt.bar(range(len(topratings.keys())),topratings.values())
    plt.xticks(range(len(topratings.keys())), topratings.keys(),rotation=70)
    plt.xlabel('Ratings')
    plt.ylabel('Number of talks with the highest rating')
    plt.tight_layout()
    plt.savefig(outfolder+'toprating_count.eps')

    # Plot the total number of talks with some specific ratings
    plt.figure(4)
    totalrat = {key:sum(vals) for key,vals in ratinglst.items()}
    plt.bar(range(len(totalrat.keys())),totalrat.values())
    plt.xticks(range(len(totalrat.keys())), totalrat.keys(),rotation=70)
    plt.xlabel('Ratings')
    plt.ylabel('Count of the ratings in all talks')
    plt.tight_layout()
    plt.savefig(outfolder+'totalrating_barplot.eps')

    # Plot number of individual ratings
    for i,akey in enumerate(ratinglst):
        plt.figure(5+i)
        plt.hist(ratinglst[akey],bins=25)
        plt.xlabel('Bins with percentage of the ratings')
        plt.ylabel('Number of talks falling in a specific bin')
        plt.title('Histogram for the rating: '+akey)
        plt.tight_layout()
        plt.savefig(outfolder+'ratings_hist_'+akey+'.eps')

if __name__=='__main__':
    infolder = os.path.join(ted_data_path,'talks/')
    outfolder = os.path.join(ted_data_path,'TED_stats/')
    print 'Input Folder = ',infolder
    print 'Output Folder = ',outfolder
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plot_statistics(infolder,outfolder)
