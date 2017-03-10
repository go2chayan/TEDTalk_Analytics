import os
import re
import cPickle as cp
import datetime

'''
This program creates an index with the  most relevant information 
from the pkl files. It is troublesome to iterate through the pkl
files for viewing simple information (like total view count etc.)
from the whole database. For that reason, this index is created.
It also outputs a list of all the tags (sounds etc. put into a parentheses)
'''

# Location where the .pkl files reside
talkspath = './talks/'
# Output file
outfile = './index.csv'
# Output Tag list
outtagfile = './tag_list.txt'

ratinglist = ['beautiful','funny', 'ingenious', 'ok', 'fascinating',\
 'total_count', 'persuasive', 'inspiring', 'longwinded', 'informative',\
  'jaw-dropping', 'obnoxious', 'confusing', 'courageous', 'unconvincing']

allfiles = [afile for afile in os.listdir(talkspath) if afile.endswith('pkl')]
alltags = []
with open(outfile,'wb') as ofile:
    # Write header
    ofile.write('Video_ID,Title,'+','.join(ratinglist)+
        ',Totalviews,Retention_time_in_days,Video_Length_(sec),'+
        'Nb_Words_in_Transcript_(without_tags),Words_per_second,Keywords,Is_a_Talk?\n')
    # Scan through the folder
    for afile in allfiles:
        print talkspath+afile
        data = cp.load(open(talkspath+afile,'rb'))
        # ID
        ofile.write(afile.split('.')[0]+',')
        # Title
        ofile.write(data['talk_meta']['title'].replace(',',' ')+',')
        # Ratings
        ratings = [str(data['talk_meta']['ratings'][akey]) \
            for akey in ratinglist]
        ofile.write(','.join(ratings)+',')
        # Total views
        ofile.write(str(data['talk_meta']['totalviews'])+',')
        # Retention time in days
        x=data['talk_meta']['datecrawled']-data['talk_meta']['datepublished']
        ofile.write(str(x.days)+',')
        # Video Length in seconds
        ofile.write(str(data['talk_meta']['vidlen'])+',')
        # Find all the tags 
        txt = ' '.join(data['talk_transcript'])
        alltags.extend(re.findall('\([a-zA-Z]*?\)',txt))
        # Number of words in the transcript excluding tags
        wrd_count=len(re.sub('\([a-zA-Z]*?\)','',txt).split())
        ofile.write(str(wrd_count)+',')
        # Words per second
        wps = float(wrd_count)/data['talk_meta']['vidlen']
        ofile.write(str(wps)+',')
        # Keywords
        ofile.write(';'.join(data['talk_meta']['keywords'])+',')
        # A heuristic estimation of if it is actually a public speech
        if 'live music' in data['talk_meta']['keywords']:
            ofile.write('No'+'\n')
        elif wrd_count <= 450 and (wps < 1. or any(item in \
                data['talk_meta']['keywords'] for item in \
                ['live music','dance','music','performance','entertainment'])):
            ofile.write('No'+'\n')
        else:
            ofile.write('Yes'+'\n')

with open(outtagfile,'wb') as tfile:
    tfile.write('\n'.join(list(set(alltags))))


        