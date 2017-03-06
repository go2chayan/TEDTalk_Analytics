import json
import os
import re
import cPickle as cp
import numpy as np
from nltk.tokenize import sent_tokenize
from watson_developer_cloud import ToneAnalyzerV3

imporant_talks=[66,96,97,206,229,549,618,
                685,741,848,1246,1344,1377,1569,1647,1815,1821,
                2034,2399,2405,220,268,339,345,379,402,403,427,
                439,500,673,675,679,925,962,1294,1332,1373,
                1445,1466]

# Use the bluemix api to extract tones
def fetch_partial_annotations():
    alltalks = [afile[:-4] for afile in os.listdir('./talks/') if afile.endswith('.pkl')]
    skipped_talks = [afile[:-4] for afile in os.listdir('./bluemix_sentiment_partial/') if afile.endswith('.pkl')]

    tone_analyzer = ToneAnalyzerV3(
       username='3b3ee4b8-7096-4cc6-89ed-119f71d49479',
       password='KOzTz2q5c6ot',
       version='2016-05-19 ')

    for atalk in alltalks:
        if atalk in skipped_talks:
            print 'skipping:',atalk
            continue
        filename = './talks/'+atalk+'.pkl'
        print filename
        data = cp.load(open(filename))
        txt = re.sub('\([a-zA-Z]*?\)','',' '.join(data['talk_transcript']))
        response = tone_analyzer.tone(text=txt)
        print response
        with open('./bluemix_sentiment/'+atalk+'.pkl','wb') as f:
            cp.dump(response,f)

# Bluemax gives tone only for the first 100 sententences. This function gets the remaining annotations
def fetch_remaining_annotations():
    imporant_talks=[66,96,97,206,229,549,618,
                685,741,848,1246,1344,1377,1569,1647,1815,1821,
                2034,2399,2405,220,268,339,345,379,402,403,427,
                439,500,673,675,679,925,962,1294,1332,1373,
                1445,1466]
    alltalks = [afile[:-4] for afile in os.listdir('./talks/') if afile.endswith('.pkl')]
    skipped_talks = [afile[:-4] for afile in os.listdir('./bluemix_sentiment/') if afile.endswith('.pkl')]

    tone_analyzer = ToneAnalyzerV3(
       username='3b3ee4b8-7096-4cc6-89ed-119f7i1d49479',
       password='KOzTz2q5c6ot',
       version='2016-05-19 ')

    for atalk in imporant_talks:
        if str(atalk) in skipped_talks:
            print 'skipping:',atalk
            continue
        filename = './talks/'+atalk+'.pkl'
        print filename
        data = cp.load(open(filename))
        txt = re.sub('\([a-zA-Z]*?\)','',' '.join(data['talk_transcript']))
        sententences = sent_tokenize(txt)
        m = len(sententences)
        if m<=100:
            continue
        else:
            pass
            # TODO: Finish it





def parse_tone_categories(categ_list):
    header=[]
    scores=[]
    for acat in categ_list:
        for atone in acat['tones']:
            header.append(acat['category_id']+'_'+atone['tone_id'])
            scores.append(atone['score'])
    return header,scores

def parse_sentence_tone(senttone_list):
    frm_idx=[]
    to_idx=[]
    sentences=[]
    header=[]
    scores=[]
    for asent in senttone_list:
        frm_idx.append(asent['input_from'])
        to_idx.append(asent['input_to'])
        sentences.append(asent['text'])
        if asent['sentence_id']==0:
            header,score = parse_tone_categories(asent['tone_categories'])
        else:
            _,score = parse_tone_categories(asent['tone_categories'])
        scores.append(score)
    scores = np.array(scores)
    return scores,header,sentences,frm_idx,to_idx





