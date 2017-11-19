import os
import re
import shutil
import cPickle as cp
import numpy as np
import math
from multiprocessing import Process
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from bluemix_key import *
from TED_data_location import ted_data_path
from list_of_talks import all_valid_talks

'''
This module extracts the bluemix scores from IBM Watson Tone Analyzer.
The code and its core assumptions are altered on October 30th to make
it consistent with the new crawler format and the overall folder
structure. The code is employed to extract the bluemix scores for the
new TED talks.
Please note that this module assumes the existence of a working
credential in the bluemix_key file.
'''

# Use the bluemix api to extract tones
def fetch_partial_annotations(startidx,endidx):
    # Create all paths
    metafolder = os.path.join(ted_data_path,'talks/')
    outfolder = os.path.join(ted_data_path,\
            'bluemix_sentiment/')
    partfolder = os.path.join(ted_data_path,'bluemix_sentiment_partial/') 
    if not os.path.exists(partfolder):
        os.mkdir(partfolder)
    # List existing full and partial data
    full_score = [int(afile[:-4]) for afile in \
        os.listdir(outfolder) if afile.endswith('.pkl')]
    part_score = [int(afile[:-4]) for afile in \
        os.listdir(partfolder) if afile.endswith('.pkl')]

    # Start processing
    for atalk in all_valid_talks:
        if atalk<startidx or atalk>endidx or  atalk in full_score\
                or atalk in part_score:
            print 'skipping:',atalk
            continue
        filename = os.path.join(metafolder,str(atalk)+'.pkl')
        print filename
        data = cp.load(open(filename))
        txt = ' '.join([aline.encode('ascii','ignore') for apara \
                in data['talk_transcript'] for aline in apara])
        # remove tags
        txt = re.sub('\([\w ]*?\)','',txt)
        response = tone_analyzer.tone(text=txt)
        print response
        with open(os.path.join(partfolder,str(atalk)+'.pkl'),'wb') as f:
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
        if not score:
            continue
        scores.append(score)
    scores = np.array(scores)
    return scores,header,sentences,frm_idx,to_idx

# Bluemax gives tone only for the first 100 sentences. 
# This function gets the remaining annotations
def fetch_remaining_annotations(startidx,endidx,
        talksdir='talks/',
        outdir='bluemix_sentiment/',
        partdir='bluemix_sentiment_partial/'):
    # Create all paths
    metafolder = os.path.join(ted_data_path,talksdir)
    outfolder = os.path.join(ted_data_path,outdir)
    partfolder = os.path.join(ted_data_path,partdir) 
    if not os.path.exists(partfolder):
        os.mkdir(partfolder)
    # List existing full and partial data
    full_score = [int(afile[:-4]) for afile in \
        os.listdir(outfolder) if afile.endswith('.pkl')]
    part_score = [int(afile[:-4]) for afile in \
        os.listdir(partfolder) if afile.endswith('.pkl')]

    for atalk in all_valid_talks:
        if atalk in full_score or atalk not in part_score:
            print 'skipping:',atalk
            continue
        print atalk
        # Source and destination files
        src = os.path.join(partfolder,str(atalk)+'.pkl')
        dst = os.path.join(outfolder,str(atalk)+'.pkl')

        # Read the current file and check transcript length
        filename = os.path.join(metafolder,str(atalk)+'.pkl')
        data = cp.load(open(filename))
        txt = ' '.join([aline.encode('ascii','ignore') for apar in\
                data['talk_transcript'] for aline in apar])
        # remove tags
        txt = re.sub('\([\w ]*?\)','',txt)
        sentences = sent_tokenize(txt)
        if len(sentences)<=100:
            # Old annotation has all the information. So skip.
            print 'Less than 100 sentences. copying directly',atalk
            shutil.copyfile(src,dst)
            continue
        else:
            # This is the partial score data
            existingdata = cp.load(open(src))
            # Mark pickles without sentence-wise score
            if not existingdata.get('sentences_tone'):
                print 'Sentence-wise annotation not found.'\
                        ' Marking it and skipping ...'
                shutil.copyfile(src,os.path.join(outfolder,str(atalk)+\
                        '_no_sentence.pkl'))
                continue
            # Processing sentence-wise scores (adding missing scores)
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
            cp.dump(existingdata,open(dst,'wb'))

def pipeline(st,en):
    fetch_partial_annotations(st,en)
    fetch_remaining_annotations(st,en)


if __name__=='__main__':
    p1 = Process(target=pipeline,args=(1,725))
    p1.start()
    p2 = Process(target=pipeline,args=(725,1450))
    p2.start()
    p3 = Process(target=pipeline,args=(1450,2175))
    p3.start()
    p4 = Process(target=pipeline,args=(2175,2900))
    p4.start() 



