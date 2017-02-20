import cPickle as cp
import os
import nltk
import matplotlib.pyplot as plt

alltalks = [afile for afile in os.listdir('./talks/') if afile.endswith('.pkl')]
tottalks = len(alltalks)
totlen,totut,tottok,totsent = 0,0,0,0
lenlst,viewlst,ratinglst,topratings,timealive,kwlst=[],[],{},{},[],[]
for afile in alltalks:
    print afile
    atalk=cp.load(open('./talks/'+afile,'rb'))
    # Length of video
    vidlength = atalk['talk_meta']['vidlen']
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
    dtdelta = atalk['talk_meta']['datecrawled'] - atalk['talk_meta']['datepublished']
    timealive.append(dtdelta.days)
    # Total number of utterances
    totut +=len(atalk['talk_transcript'])
    # Total number of words and sentences
    fulltrns = ' '.join(atalk['talk_transcript'])
    totsent+= len(nltk.sent_tokenize(fulltrns))
    tottok += len(nltk.word_tokenize(fulltrns))

print 'Number of talks:',tottalks
print 'Total length of the talks:',totlen,'seconds'
print 'Total number of utterances:',totut
print 'Total number of words:',tottok
print 'Total number of sentences:',totsent

print 'Average words per talk:', float(tottok)/float(tottalks)
print 'Average sentences per talk:', float(totsent)/float(tottalks)
print 'Average words per sentence:', float(tottok)/float(totsent)
print 'Average length of talks:', float(totlen)/float(tottalks),'seconds'

# Plot video-length distribution
plt.figure(1)
plt.hist(lenlst,bins=50)
plt.xlabel('Durations of the videos (sec)')
plt.ylabel('Frequency')
plt.savefig('duration_hist.pdf')

# Plot viewcount distribution
plt.figure(2)
plt.hist(viewlst,bins=50)
plt.xlabel('Number of views in a video')
plt.ylabel('Frequency')
plt.savefig('viewcount_hist.pdf')

# Plot number of talks with some specific ratings as maximum
plt.figure(3)
plt.bar(range(len(topratings.keys())),topratings.values())
plt.xticks(range(len(topratings.keys())), topratings.keys(),rotation=70)
plt.xlabel('Various Ratings')
plt.ylabel('Number of talks having a specific rating higher than all other ratings')
plt.tight_layout()
plt.savefig('toprating_count.pdf')

# Plot the total number of talks with some specific ratings
plt.figure(4)
totalrat = {key:sum(vals) for key,vals in ratinglst.items()}
plt.bar(range(len(totalrat.keys())),totalrat.values())
plt.xticks(range(len(totalrat.keys())), totalrat.keys(),rotation=70)
plt.xlabel('Various Ratings')
plt.ylabel('Total count of the rating in all talks')
plt.tight_layout()
plt.savefig('totalrating_barplot.pdf')


# Plot number of individual ratings
for i,akey in enumerate(ratinglst):
    plt.figure(5+i)
    plt.hist(ratinglst[akey],bins=25)
    plt.xlabel('Bins with percentage of the ratings')
    plt.ylabel('Number of talks falling in a specific bin')
    plt.title('Histogram for the rating: '+akey)
    plt.tight_layout()
    plt.savefig('ratings_hist_'+akey+'.pdf')






