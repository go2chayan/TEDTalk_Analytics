import os
import re
import csv
import urllib2
import sys
import json
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from time import sleep
import cPickle as cp
from subprocess import call
from TED_data_location import ted_data_path

"""
The older crawler is not usable because the TED talk website is
changed recently (As seen in Oct 20th, 2017). Therefore, we need a 
new crawler. In the new system, the transcripts are timestamped
per paragraph, not per utterance. Also, the meta data contains
an additional JSON containing the complete meta data. Other
aspects are tried to keep backward compatible.

The crawler automatically downloads the videos unlike the previous crawler
"""

def request_http(url):
    count = 0
    print 'requested:',url
    sys.stdout.flush()
    text_seg=None
    while count < 100:
        # sleep 5 seconds
        sleep(2)
        try:
            resp = urllib2.urlopen(url)
            break
        except urllib2.HTTPError as e:
            if e.code == 404:
                raise
            else:
                count+=1
                print 'HTTP Error code:',e.code
                print 'HTTP Error msg:',e.msg
                print 'Too frequent HTTP call (',count,') ... sleeping ...'
                # Random waiting up to 60 sec
                sleep(int(np.random.rand(1)[0]*60))
                print 'Trying again ...'
                sys.stdout.flush()
                continue

    web_src = resp.read().decode('utf8','ignore').replace('\r',' ').replace('\n', ' ')
    text_seg = BeautifulSoup(web_src, 'lxml')
    if not text_seg:
        raise IOError('HTTP Failure')
    return text_seg

def get_trans_new(src_url):
    '''
    Get the transcripts from the new format (as of Aug 16, 2017) of the
    TED talk web pages.
    '''
    talk_text = ''
    text_seg = request_http(src_url+'/transcript/transcript?language=en')
    time_divs = text_seg.find_all('div',
            {'class':' Grid__cell w:1of6 w:1of8@xs w:1of10@sm w:1of12@md '})
    text_divs = text_seg.find_all('div',
     {'class':' Grid__cell w:5of6 w:7of8@xs w:9of10@sm w:11of12@md p-r:4 '})
    # Store the time
    trns_micsec = []
    for atime in time_divs:
        mins,secs = atime.contents[1].contents[0].strip().split(':')
        trns_micsec.append((int(mins)*60+int(secs))*1000)
    # Store the text
    trns_text=[]
    for atext in text_divs:
        trns_text.append([aspan.strip() for aspan in re.split('\t*',
            atext.contents[1].contents[0])
            if aspan.strip()])
    if not trns_text or not trns_micsec:
        raise IOError('Empty transcripts')
    return trns_text,trns_micsec

def get_meta_new(url_link):
    '''
    This is the function to extract the meta information from
    the new format (as of Oct 20, 2017) of TED talk web pages.
    '''
    # Retrieve and parse html
    text_seg = request_http(url_link)

    # Identify correct block for next piece of information
    scripts = text_seg.find_all('script')
    for ascript in scripts:
        if not ascript.getText().startswith('q("talkPage.init"'):
            continue
        # Get the JSON containing information about the talk
        fullJSON =json.loads(re.search('(?<=q\(\"talkPage\.init\"\,\s)(\{.*\})',
            ascript.contents[0]).group(0))['__INITIAL_DATA__']
        # ID of the current talk
        talk_id = fullJSON['current_talk']
        currenttalk_JSON=None
        # Identify the JSON part for the current talk
        for atalk in fullJSON['talks']:
            if not atalk['id'] == talk_id:
                continue
            else:
                currenttalk_JSON = atalk
                break
        # Make sure that currenttalk_JSON is not none
        assert currenttalk_JSON, IOError('JSON detail of the talk is not found')

        ################## Extract all the necessary components ################
        # Get title
        title = currenttalk_JSON['title']
        # Get Author
        author=''
        for a_speaker in currenttalk_JSON['speakers']:
            author = author+a_speaker['firstname']+'_'+a_speaker['lastname']+';'
        # Get Keyword
        keywrds = currenttalk_JSON['tags']
        # Duration
        vidlen = currenttalk_JSON['duration']
        # Get the ratings as JSON string
        ratingJSON = currenttalk_JSON['ratings']
        ratings={}
        totcount=0
        for item in ratingJSON:
            ratings[str(item['name']).lower()]=item['count']
            totcount+=item['count']
        ratings['total_count']=totcount
        # Date Crawled
        datecrawl = datetime.now()
        # Download link
        if 'media' in fullJSON and 'internal' in fullJSON['media'] and \
                'podcast-regular' in fullJSON['media']['internal']:
            downlink = fullJSON['media']['internal']['podcast-regular']['uri']
        elif 'media' in fullJSON and 'internal' in fullJSON['media'] and \
            len(fullJSON['media']['internal'].keys()) > 0:
            # If the regular podcast link is not available
            # save whatever is available
            linktype = fullJSON['media']['internal'].keys()[0]
            downlink = fullJSON['media']['internal'][linktype]['uri']
        else:
            downlink=''
        # Date published and Date Filmed
        for player_talk in currenttalk_JSON['player_talks']:
            datepub=-1
            datefilm=-1
            if player_talk['id']==talk_id:
                datepub = player_talk['published']
                datefilm = player_talk['published']
                break
        assert datepub is not -1 and datefilm is not -1,'Could not extract datepub or datefilm'
        # datepub = np.datetime64(
        #         currenttalk_JSON['speakers'][0]['published_at']).astype('O')
        # datefilm = np.datetime64(currenttalk_JSON['recorded_at']).astype('O')
        datepub = datetime.fromtimestamp(datepub)
        datefilm = datetime.fromtimestamp(datefilm)
         # Total views
        totviews = fullJSON['viewed_count']
        #########################################################################
        break
    return {'ratings':ratings,'title':title,'author':author,'keywords':keywrds,
    'totalviews':totviews,'downloadlink':downlink,'datepublished':datepub,
    'datefilmed':datefilm,'datecrawled':datecrawl,'vidlen':vidlen,'id':int(talk_id),
    'alldata_JSON':json.dumps(fullJSON),'url':url_link}

def crawl_and_update(csvfilename,
        videofolder,
        outfolder,
        split_idx=-1,
        split_num=-1):
    '''
    Crawls the TED talks and extracts the relevant information.
    '''
    # Talk ID's to skip
    if os.path.isfile('to_skip.txt'):
        with open('to_skip.txt','rb') as f:
            toskip=[int(id) for id in f]
    # Build a list of urls to skip: all successes and failures
    # This is to skip a talk without actually visiting them
    toskip_url=[]
    if os.path.isfile('./success.txt'):
        with open('./success.txt') as f:
            toskip_url.extend([aurl.strip() for aurl in f])
    if os.path.isfile('./failed.txt'):
        with open('./failed.txt') as f:
            toskip_url.extend([aurl.strip() for aurl in f])
    toskip=set(toskip)
    toskip_url=set(toskip_url)
    # Debug
    print 'Opening the csv file'
    sys.stdout.flush()
    # New style csv file
    with open(csvfilename,'rU') as f:
        line_num = sum([1 for arow in csv.DictReader(f)])
    with open(csvfilename,'rU') as f:
        csvfile = csv.DictReader(f)
        # debug
        print 'csv file opened successfully'
        sys.stdout.flush()
        # Starting to read the csv file
        for rownum,arow in enumerate(csvfile):
            if split_idx is not -1:
                datslice = line_num/split_num
                if rownum < split_idx*datslice or rownum >= (split_idx+1)*datslice:
                    continue
            print 'split_idx =',split_idx
            print 'current row =',rownum
            sys.stdout.flush()
            url = arow['public_url']
            # Skip if already tried (succeded or failed)
            if url.strip() in toskip_url:
                continue
            ######################### Get Meta ############################
            try:
                meta = get_meta_new(url)
            except Exception as e__:
                print
                print e__
                print 'Failed to extract meta. continuing'
                sys.stdout.flush()
                # No meta means a failure
                with open('./failed.txt','a') as ferr:
                    ferr.write(url+'\n')
                continue
            # Meta successfully extracted
            id_ = meta['id']
            print 'examining ...',id_,url
            sys.stdout.flush()
            # Skip if it is supposed to skip
            if id_ in toskip:
                print '... skipping'
                sys.stdout.flush()
                continue
            target_filename = os.path.join(outfolder,str(id_)+'.pkl')
            ########################## Get Transcript #######################
            try:
                txt,micstime = get_trans_new(url)
            except Exception as e:
                print
                print e
                print 'Transcript not found for,',id_
                sys.stdout.flush()
                # Not being able to find transcript means a failure
                with open('./failed.txt','a') as ferr:
                    ferr.write(url+'\n')
                continue
            ########################## Save Everything ######################
            cp.dump({'talk_transcript':txt,'transcript_micsec':micstime,
                    'talk_meta':meta},open(target_filename,'wb'))
            # Now save the video
            target_videofile = os.path.join(videofolder,str(id_)+'.mp4')
            if os.path.exists(target_videofile):
                print 'Video exists. skipping ...'
                # Record Successes
                with open('./success.txt','a') as fsucc:
                    fsucc.write(url+'\n')
                sys.stdout.flush()
                continue
            print 'Video downloader started'
            sys.stdout.flush()
            if meta['downloadlink']:
                call(['wget','-O',target_videofile,meta['downloadlink']])
            else:
                print 'Video could not save. No link found',id_
                sys.stdout.flush()
            # Record Successes
            with open('./success.txt','a') as fsucc:
                fsucc.write(url+'\n')

if __name__=='__main__':
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        print 'SLURM_ARRAY_TASK_ID=',os.environ['SLURM_ARRAY_TASK_ID']
        sys.stdout.flush()
        crawl_and_update(
            './TED Talks as of 08.04.2017.csv',
            os.path.join(ted_data_path,'TED_video/'),
            os.path.join(ted_data_path,'TED_meta/'),
            split_idx=int(os.environ['SLURM_ARRAY_TASK_ID']),
            split_num=int(os.environ['TASK_SPLIT']))
    else:
        print 'SLURM ID not found'
        sys.stdout.flush()
        crawl_and_update(
            './TED Talks as of 08.04.2017.csv',
            os.path.join(ted_data_path,'TED_video/'),
            os.path.join(ted_data_path,'TED_meta/'))


