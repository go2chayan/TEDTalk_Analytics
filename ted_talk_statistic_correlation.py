import cPickle as cp
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np

# Returns indices with outliers removed
def remove_outlier(alist):
    nplist = np.array(alist)
    mean = np.mean(nplist)
    std = np.std(nplist)
    idx = np.where((nplist<mean+3.*std) & (nplist>mean-3.*std))
    print 'Outlier Removal'
    print 'input list length:',len(alist)
    print 'output list length:',len(idx[0])
    print 'outliers:',len(alist)-len(idx[0])
    print
    return idx[0]

def plot_correlation(infolder='./talks/',outfolder='./plots/'):
    alltalks = [afile for afile in os.listdir(infolder) if afile.endswith('.pkl')]
    tottalks = len(alltalks)
    totlen,totut,tottok,totsent = 0,0,0,0
    lenlst,viewlst,ratinglst,topratings,timealive,kwlst=[],[],{},{},[],[]
    titles=[]
    allratings={}
    allrating_names=['beautiful','confusing','courageous','fascinating','funny',\
            'informative','ingenious','inspiring','jaw-dropping','longwinded',\
            'obnoxious','ok','persuasive','total_count','unconvincing']
    # Reading all the pickle files and enlisting required info
    for afile in alltalks:
        print afile
        atalk=cp.load(open(infolder+afile,'rb'))
        # View count
        viewlst.append(atalk['talk_meta']['totalviews'])
        # Update total ratings and list the highest rating of each talk
        for akey in allrating_names:
            if akey=='total_count':
                continue
            if not allratings.get(akey):
                allratings[akey]=[atalk['talk_meta']['ratings'].get(akey,0)]
            else:
                allratings[akey].append(float(atalk['talk_meta']['ratings'].get(akey,0))/\
                    float(atalk['talk_meta']['ratings']['total_count']))
    # Drawing the scatter plots
    for ind,akey in enumerate(allratings):
        plt.figure(ind)
        # remove the outliers because some ratings are so high that it skews the plot
        idx = remove_outlier(allratings[akey])
        x = [viewlst[i] for i in idx]
        y = [allratings[akey][i] for i in idx]
        # Calculate Correlation Coefficient
        z = np.corrcoef(x,y)[0,1]
        print 'Correlation coefficient for rating',akey,'and view count:',z
        plt.scatter(x,y)
        # plot the x axis in log scale
        plt.xscale('log',nonposy='clip')
        plt.xlabel('Total Viewcount (log scale)')
        plt.ylabel('Percent of ratings')
        plt.title('rating-view scatter plot '+akey+'corr: {:0.2f}'.format(z))
        plt.tight_layout()
        plt.savefig(outfolder+'scatter_'+akey+'.pdf')

if __name__=='__main__':
    plot_correlation()