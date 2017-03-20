from ted_talk_experiments import time_avg_hi_lo_ratings
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import random


titles = "anger,disgust,fear,joy,sadness,analytical,confident,tentative,openness,conscientiousness,extraversion,agreeableness,emotion"
lab = ["anger", "disgust", "fear", "joy", "sadness", "analytical", "confident", "tentative", "openness", "conscientiousness", "extraversion", "agreeableness", "emotion"]
ratings = ["Beautiful","Funny","Ingenious","OK","Fascinating","Persuasive","Inspiring", "Longwinded","Informative","JawDropping","Obnoxious","Confusing","Courageous","Unconvinving"]    
top = ['High_Funny_Percent', 'Low_Funny_Percent', 'High_Fascinating_Percent', 'Low_Fascinating_Percent','High_Jaw_Dropping_Percent','Low_Jaw_Dropping_Percent', 'High_Inspiring_Percent','Low_Inspiring_Percent']
mid = ['High_Persuasive_Percent', 'Low_Persuasive_Percent', 'High_Courageous_Percent', 'Low_Courageous_Percent','High_Beautiful_Percent','Low_Beautiful_Percent', 'High_Ingenious_Percent','Low_Ingenious_Percent', 'High_Informative_Percent', 'Low_Informative_Percent', 'High_ok_Percent','Low_ok_Percent']
last = ['High_Unconvincing_Percent', 'Low_Unconvincing_Percent', 'Low_Confusing_Percent','High_Confusing_Percent', 'High_Obnoxious_Percent', 'Low_Obnoxious_Percent', 'High_Longwinded_Percent', 'Low_Longwinded_Percent']
allr = top + mid + last
axes_values = []
for x in titles.split(","):
    axes_values.append(x)


class Radar(object):

    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles.split(',')) 
        self.angles = np.arange(0, 360, 360.0/self.n)
        print axes_values
        for i in range(0,self.n):
            ll = axes_values[i]
            print ll
        self.axes = [fig.add_axes(rect, projection="polar", label= axes_values[i]) 
                         for i in range(0,self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=lab, fontsize=10)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            #ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.set_rgrids([0.2,0.4,0.6,0.8,1.0, 1.2,1.4,1.6], angle=angle, labels=label, fontsize=10)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 1.6)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


def plotradar(avg_saved, ratingList, name):
    fig = pl.figure(figsize=(11, 13))
    labels = [["0.2","0.4", "0.6", "0.8", "1.0", "1.2","1.4","1.6"], list(""), list(""),list(""),list(""),list(""), list(""), list(""),list(""),list(""),list(""), list(""), list("")]
    radar = Radar(fig, titles, labels)
    clr = ['#0000ff', '#cc0066', '#009933', '#9900cc', '#009999','#006666','#0099cc','#ff9966','#666699','#666633','#ffff00','#ff0000','#666633','#003300'] 
    iii =0
    for i in range(0,14):
        print i
        hi = []
        lo = []
        if((avg_saved[i].items()[0][0] not in ratingList)):
            continue;

        if("High" in avg_saved[i].items()[0][0]):
            for ii in range(0,13):
                hi.append(avg_saved[i].items()[0][1][ii])
                lo.append(avg_saved[i].items()[1][1][ii]) 
        else:
            for ii in range(0,13):
                lo.append(avg_saved[i].items()[0][1][ii])
                hi.append(avg_saved[i].items()[1][1][ii])
        rationvalue = (np.divide(hi, lo))
        print rationvalue
        c = clr[iii]
        iii = iii + 1
        radar.plot(rationvalue,  "-", lw=2, color=c, marker = "o",alpha=1.0, label=ratings[i], linewidth = .7)
    radar.ax.legend(loc='upper right',fancybox=True, framealpha=0.4, prop={'size':10})
    filenm = name+'.png'
    fig.savefig(filenm)

avg_saved = time_avg_hi_lo_ratings()
plotradar(avg_saved, top, "top4RatingVSsentiment")
plotradar(avg_saved, mid, "mid6RatingVSsentiment")
plotradar(avg_saved, last, "last4RatingVSsentiment")
plotradar(avg_saved, allr, "all14RatingVSsentiment")