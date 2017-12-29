import csv
import itertools
import operator as op
from list_of_talks import all_valid_talks
from ted_talk_sentiment import Sentiment_Comparator, read_bluemix
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway,ttest_ind

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
    only one sentiment score. Practically it is equivalent to considering that 
    X_1 is of order 2 (NxM), instead of 3 (NxMxB). In addition, it performs 
    z-score standardization of the rows of X_1 (i.e. each talk).
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

def clust_separate_stand(X,clusterer,comparator,csvcontent,csv_vid_idx):
    '''
    Cluster the videos for each individual score. Notice that it 
    formulates different clusters while considering different scores.
    Although it is a bit slow due to some recomputations, but this 
    would give better results in the clustering. Also, it z-score 
    standardizes each TED talks signal which would reveal the 
    storytelling patterns better.
    '''
    N,M,B = X.shape
    avg_dict = {}
    for s in range(B):
        # Perform clustering over each score
        clust_dict = clust_onescore_stand(X[:,:,s],clusterer,comparator)
        comparator.reform_groups(clust_dict)
        avg = comparator.calc_group_mean()
        # Although it computed the average for all the columns, I need
        # just one, s'th column. This is the recomputation. I don't
        # think it is too bad, though.
        print
        print
        print 'Clustering for:',comparator.column_names[s]
        print '================================'        
        for aclust in avg:
            if not comparator.column_names[s] in avg_dict:
                avg_dict[comparator.column_names[s]] = {aclust:avg[aclust][:,s]}
            else:
                avg_dict[comparator.column_names[s]][aclust]=avg[aclust][:,s]
            # Print information about this cluster
            totview=[]
            for vid in clust_dict[aclust]:
                i = csv_vid_idx[vid]
                totview.append(int(csvcontent['Totalviews'][i]))
                #totview.append(int(csvcontent['beautiful'][i]))
            print aclust+':'
            print '----------------'
            print 'Average View Count:',np.mean(totview)
    return avg_dict

def evaluate_clust_separate_stand(X,clusterer,comparator,\
    csvcontent,csv_id,b_=None,outfilename=None):
    '''
    It is similar to clust_separate_stand, but instead of returning
    a dictionary, it draws the cluster means and evaluate the differences
    in various clusters. It performs ANOVA to check if the 
    clusters have any differences in their ratings
    Edit: Now it also performs (Based on CHI Reviewer's recommendations)
    1. ANOVA with Bonferroni correction
    2. Pairwise multiple t-test with Bonferroni correction
    3. Effectsize and direction of the clusters on the ratings
    '''
    N,M,B = X.shape
    avg_dict = {}
    kwlist = ['beautiful', 'ingenious', 'fascinating',
                'obnoxious', 'confusing', 'funny', 'inspiring',
                 'courageous', 'ok', 'persuasive', 'longwinded', 
                 'informative', 'jaw-dropping', 'unconvincing','Totalviews']
    plt.close('all')
    # s is the index of a bluemix score
    for s in range(B):
        # If b_ is specified, just compute one score and skip others
        if b_ and not b_ == s:
            continue
        # Perform clustering over each score
        clust_dict = clust_onescore_stand(X[:,:,s],clusterer,comparator)
        comparator.reform_groups(clust_dict)
        avg = comparator.calc_group_mean()
        for aclust in avg:
            if not comparator.column_names[s] in avg_dict:
                avg_dict[comparator.column_names[s]] = {aclust:avg[aclust][:,s]}
            else:
                avg_dict[comparator.column_names[s]][aclust]=avg[aclust][:,s]
        # Pretty draw the clusters
        draw_clusters_pretty(avg_dict,comparator,csvcontent,csv_id,
            b_=s,outfilename=outfilename)
        
        # Now apply ANOVA and compare clusters
        pvals = {}
        allvals = {}
        # Formulate a list of values for each rating
        print '='*50
        print '{:^50}'.format('HYPOTHESIS TESTS')
        print '{:^50}'.format('for IBM Score:'+comparator.column_names[s])
        print '='*50
        for akw in kwlist:
            if akw == 'Totalviews':
                ratvals = {aclust:[int(csvcontent[akw][csv_id[avid]]) for avid\
                    in comparator.groups[aclust]] for aclust in \
                    comparator.groups}
            else:
                ratvals = {aclust:[float(csvcontent[akw][csv_id[avid]])/\
                        float(csvcontent['total_count'][csv_id[avid]])\
                        for avid in comparator.groups[aclust]] for\
                        aclust in comparator.groups}
            #################### perform ANOVA #####################
            ratval_itemlist = list(zip(*ratvals.items())[1])
            _,pval = f_oneway(*ratval_itemlist)
            # Save only the statistically significant ones
            if pval<0.05:
                print 'ANOVA p value ('+akw+'):',pval
                print 'ANOVA p value ('+akw+') with Bonferroni:',\
                    pval*float(len(kwlist)),
                if pval*float(len(kwlist)) < 0.05:
                    print '< 0.05'
                    pvals[akw]=pval*float(len(kwlist))
                    allvals[akw] = ratval_itemlist
                else:
                    print 'not significant'
            ########### Pair-wise t-test with correction ###########
            # Total number of repeated comparisons
            paircount = count_n_choose_r(len(ratvals),2)
            # Pair-wise comparison using t-test and effectsize
            for rat1,rat2 in itertools.combinations(ratvals,2):
                _,pval_t = ttest_ind(ratvals[rat1],ratvals[rat2],\
                    equal_var=False)
                # Perform Bonferroni Correction for multiple t-test
                pval_t = pval_t*float(paircount)
                # Check significance
                if pval_t < 0.005:
                    print 'p-val of ttest (with Bonferroni) in "'+akw+\
                        '" between '+rat1+' and '+rat2+':',pval_t
                    ############# Pair-wise Effectsizes ##############
                    n1 = len(ratvals[rat1])
                    n2 = len(ratvals[rat2])
                    sd1 = np.std(ratvals[rat1])
                    sd2 = np.std(ratvals[rat2])
                    sd_pooled = np.sqrt(((n1 - 1)*(sd1**2.) +\
                        (n2-1)*(sd2**2.))/(n1+n2-2))
                    cohen_d = (np.mean(ratvals[rat1]) - \
                        np.mean(ratvals[rat2]))/sd_pooled
                    print 'Cohen\'s d of rating "'+akw+'" between '+rat1+\
                        ' and '+rat2+': ',cohen_d
                    print



        # If the clusters are significantly different in any rating, draw it
        if not pvals.keys():
            continue
        else:
            draw_boxplots(pvals,allvals,s,comparator,outfilename=outfilename)

def draw_boxplots(pvals,allvals,s,comparator,outfilename=None):
    # Draw the box plot for Totalviews first
    for akw in pvals:
        plt.figure(comparator.column_names[s]+akw)
        ax=plt.boxplot(allvals[akw],
            labels=comparator.groups.keys(),
            showfliers=False)
        plt.ylabel('Total Views')
        plt.suptitle(\
            'Significant (p={0:0.6f}) difference in '.format(pvals[akw])+\
            akw+'\n'+'while clustering based on: '+comparator.column_names[s])
        if not outfilename:
            plt.show()
        else:
            plt.savefig(outfilename+'boxplt_'+\
                comparator.column_names[s]+'_'+akw+'.eps')
            plt.close()

def read_index(indexfile):
    # Read the content of the index file
    # content is a dictionary 
    with open(indexfile) as csvfile:
        reader=csv.DictReader(csvfile,delimiter=',')
        content={}
        vid_idx={}
        for i,arow in enumerate(reader):
            for akey in arow:
                if akey=='Video_ID':
                    vid_idx[int(arow[akey])]=i
                elif not content.get(akey):
                    content[akey]=[arow[akey]]
                else:
                    content[akey].append(arow[akey])        
        return content,vid_idx

def draw_clusters(avg_dict,column_names,fullyaxis=False,\
        outfilename=None):
    '''
    This plotter expects the avg_dict from clust_separate_stand.
    avg_dict is a dictionary containing the averages of each cluster
    '''
    for i,s in enumerate(avg_dict):
        plt.figure(figsize=(16,9))
        for akey in avg_dict[s]:
            plt.plot(avg_dict[s][akey],label=akey)
            # Print the characteristics of the cluster
            print akey
            print '============='
            print 
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
            plt.close()
    if not outfilename:
        plt.show()

def draw_clusters_pretty(avg_dict,comp,csvcontent,vid_idx,
    b_=None,outfilename=None):
    '''
    Draws the cluster means and its closest-matching talks.
    avg_dict is a dictionary containing cluster means for various scores.
    comp is the sentiment comparator object
    '''
    X = np.array([comp.sentiments_interp[atalk] for atalk in comp.alltalks])
    M = np.size(X,axis=1)
    colidx = {col:i for i,col in enumerate(comp.column_names)}
    kwlist = ['beautiful', 'ingenious', 'fascinating',
                'obnoxious', 'confusing', 'funny', 'inspiring',
                 'courageous', 'ok', 'persuasive', 'longwinded', 
                 'informative', 'jaw-dropping', 'unconvincing']
    for ascore in avg_dict:
        # b is the index of the current score
        b = colidx[ascore]
        # If b_ is specified, just draw one score and skip others
        if b_ and not b_ == b:
            continue
        # Start plotting
        fig = plt.figure(figsize=(15,7))
        nb_clust = len(avg_dict[ascore].keys())
        rows = int(np.ceil(nb_clust/3.))
        cols = 3
        print
        print
        print ascore
        print '######################'
        for c,aclust in enumerate(avg_dict[ascore]):
            # Standerdize X
            xmean = np.mean(X[:,:,b],axis=1)[None].T
            xstd = np.std(X[:,:,b],axis=1)[None].T
            Z = (X[:,:,b] - xmean)/xstd
            # Calculate the closest matches
            r = Z - avg_dict[ascore][aclust][None]
            simidx=np.argsort(np.sum(r*r,axis=1))
            yval = X[simidx[:20],:,b].T
            avg_yval = avg_dict[ascore][aclust]
            # Make the text to be shown for each cluster            
            txtlist = [csvcontent['Title'][vid_idx[comp.alltalks[idx]]]\
                for idx in simidx[:5]]
            # Print the rating averages of the clusters
            f20vids=[vid_idx[comp.alltalks[idx]] for idx in simidx[:20]]
            print
            print aclust
            print '============'
            for j,akw in enumerate(kwlist):
                amean_rat = np.mean(\
                    [float(csvcontent[akw][i])/float(csvcontent[\
                    'total_count'][i])*100 for i in f20vids])
                print 'mean rating:',akw+' : {0:2.2f}'.format(amean_rat)
            # Print the average of total view
            avview = np.mean([int(csvcontent['Totalviews'][i])\
                    for i in f20vids])
            print 'Average View: {0:0.2e}'.format(avview)
            # Draw the axes
            decorate_axis(c,cols,rows,yval,avg_yval,txtlist,aclust,fig)
        plt.suptitle(ascore.replace('_',' '))
        if not outfilename:
            plt.show()
        else:
            plt.savefig(outfilename+'clust_'+ascore+'.eps')
            plt.close()

def decorate_axis(c,cols,rows,yval,avg_yval,txtlist,legendval,fig,
        toff=0.03,boff=0.015,loff=0.02,midoff=0.03,roff=0.005,txth=0.18):
    irow = c / cols
    icol = c % cols
    cellw = (1. - loff - roff)/float(cols)
    cellh = (1. - toff - boff)/float(rows)
    axh = (cellh-midoff)/2.
    axleft = loff+icol*cellw+midoff/2.
    axbottom = boff+irow*cellh+midoff/2.+axh
    axw = cellw-midoff
    txtaxbottom = boff+irow*cellh+midoff/2.
    # Position the axes
    ax = fig.add_axes([axleft,axbottom,axw,axh])
    # Draw the average and the top 20 similar talks
    ax.plot(yval,color='gray',linewidth=0.5)
    ax.plot(avg_yval,color='orange',\
        linewidth=2,label=legendval)
    plt.ylim([0,1])
    plt.xlabel('Percent of Speech')
    plt.ylabel('Value')
    plt.legend()
    # Put the text axis
    txtax = fig.add_axes([axleft,txtaxbottom,axw,axh-toff])
    txtax.axis('off')
    txtax.patch.set_alpha(0)
    for i,txt in enumerate(txtlist):
        txtax.text(0,1 - txth*(i+1),str(i+1)+'. '+txt)

def count_n_choose_r(n,r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom
