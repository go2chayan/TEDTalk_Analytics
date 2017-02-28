import os
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
        ',Totalviews,Retention_time_in_days\n')
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
        ofile.write(str(x.days)+'\n')
        # Find all the tags 
        alltags.extend(re.findall('\([a-zA-Z]*?\)',' '.join(data['talk_transcript'])))
with open(outtagfile,'wb') as tfile:
    tfile.write('\n'.join(list(set(alltags))))


        