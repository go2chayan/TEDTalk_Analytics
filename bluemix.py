import os
import re
import cPickle as cp
import numpy as np
import math
from nltk.tokenize import sent_tokenize
from bluemix_key import *

imporant_talks=[66,96,97,206,229,549,618,
                685,741,848,1246,1344,1377,1569,1647,1815,1821,
                2034,2399,2405,220,268,339,345,379,402,403,427,
                439,500,673,675,679,925,962,1294,1332,1373,
                1445,1466]

# Use the bluemix api to extract tones
def fetch_partial_annotations():
    alltalks = [afile[:-4] for afile in os.listdir('./talks/') if afile.endswith('.pkl')]
    skipped_talks = [afile[:-4] for afile in os.listdir('./bluemix_sentiment_partial/') if afile.endswith('.pkl')]

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

# segment a list into chunks of 100's
def segment100(alist):
    m = len(alist)
    m_10 = math.ceil(float(m)/100.)*100
    segm=[alist[i:min(j,len(alist))] for i,j in \
        zip(range(0,int(m_10),100),range(100,int(m_10)+1,100))]
    return segm

# Parse the tone_categories data structure coming from bluemix
def parse_tone_categories(categ_list):
    header=[]
    scores=[]
    for acat in categ_list:
        for atone in acat['tones']:
            header.append(acat['category_id']+'_'+atone['tone_id'])
            scores.append(atone['score'])
    return header,scores

# Parse the sentences_tone data structure coming from bluemix
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

# Bluemax gives tone only for the first 100 sentences. 
# This function gets the remaining annotations
def fetch_remaining_annotations(talksdir='./talks/',
                                outdir='./bluemix_sentiment/',
                                partdir='./bluemix_sentiment_partial/'):
    imporant_talks=[1152,66,96,97,206,229,549,618,
                685,741,848,1246,1344,1377,1569,1647,1815,1821,
                2034,2399,2405,220,268,339,345,379,402,403,427,
                439,500,673,675,679,925,962,1294,1332,1373,
                1445,1466]
    alltalks = [int(afile[:-4]) for afile in os.listdir(talksdir) \
        if afile.endswith('.pkl')]
    skipped_talks = [int(afile[:-4]) for afile in os.listdir(outdir) \
        if afile.endswith('.pkl')]

    for atalk in alltalks:
        if atalk in skipped_talks:
            print 'skipping:',atalk
            continue
        filename = talksdir+str(atalk)+'.pkl'
        print filename
        data = cp.load(open(filename))
        txt = re.sub('\([a-zA-Z]*?\)','',' '.join(data['talk_transcript']))
        sentences = sent_tokenize(txt)
        if len(sentences)<=100:
            # Old annotation has all the information. So skip.
            print 'Less than 100 sentences. copying directly',atalk
            cp.dump(cp.load(open(partdir+str(atalk)+'.pkl')),\
                open(outdir+str(atalk)+'.pkl','wb'))
            continue
        else:
            # This is the old annotation
            existingdata = cp.load(open(partdir+str(atalk)+'.pkl'))
            if not existingdata.get('sentences_tone'):
                print 'Sentence-wise annotation not found. Marking it and skipping ...'
                cp.dump(cp.load(open(partdir+str(atalk)+'.pkl')),\
                    open(outdir+str(atalk)+'_no_sentence.pkl','wb'))
                continue

            old_to = existingdata['sentences_tone'][-1]['input_to']
            old_sentid = existingdata['sentences_tone'][-1]['sentence_id']
            # Segment the talk in chunks of 100 sentences
            segments = segment100(sentences)
            # Collect annotation for the rest of the talk
            for asegm in segments[1:]:
                txt = ' '.join(asegm)
                result = tone_analyzer.tone(txt)
                # Update the input_from and input_to fields accordingly
                try:
                    output = result['sentences_tone']
                except KeyError:
                    # There was only one sentence in txt
                    output=[{'input_from':0,'input_to':len(txt),'sentence_id':0,'text':txt,\
                    'tone_categories':result['document_tone']['tone_categories']}]

                for i in range(len(output)):
                    output[i]['input_from']+=old_to+1
                    output[i]['input_to']+=old_to+1
                    output[i]['sentence_id']+=old_sentid+1
                # Add the new content to existing data
                existingdata['sentences_tone'].extend(output)
                # Update old_to and old_sentid to the most recent to value
                old_to = output[i]['input_to']
                old_sentid = output[i]['sentence_id']

            cp.dump(existingdata,open(outdir+str(atalk)+'.pkl','wb'))












