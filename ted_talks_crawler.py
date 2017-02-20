import urllib2
import sys
import re
import json
import os
from bs4 import BeautifulSoup
from datetime import datetime
from time import sleep
import cPickle as cp
from random import random

def get_trans_spans(src_url):
    talk_text = ''
    resp = urllib2.urlopen(src_url+'/transcript?language=en')
    web_src = resp.read().replace('\r',' ').replace('\n', ' ')
    text_seg = BeautifulSoup(web_src, 'lxml')
    spans = text_seg.find_all('span', {'class':'talk-transcript__fragment'})
    trns_text,trns_micsec=[],[]
    for aspan in spans:
        trns_text.append(aspan.getText().replace(u'\u2014','-').encode('ascii','ignore'))
        trns_micsec.append(int(aspan.attrs['data-time']))
    return trns_text,trns_micsec

def get_meta(url_link):
    # Retrieve and parse html
    resp = urllib2.urlopen(url_link)
    web_src = resp.read().replace('\r',' ').replace('\n', ' ')
    text_seg = BeautifulSoup(web_src, 'lxml')
    # Get title
    title = str(text_seg.title.getText()).split('|')[0].split(':')[1].strip()
    # Get Author
    author = str(text_seg.find('meta',{'name':'author'}).attrs['content'])
    # Get Keyword
    keywrds = [atag.attrs['content'].lower() for atag in text_seg.find_all('meta',{'property':'video:tag'})]
    # Total views
    totviews = int(text_seg.find('span',{'class':' f:6 f-w:700 l-s:t l-h:d '}).contents[0].strip().replace(',',''))
    # find the ratings within the correct script block
    # Duration
    vidlen = int(text_seg.find('meta',{'property':'video:duration'}).attrs['content'])
    # Identify correct block for next piece of information
    scripts = text_seg.find_all('script')
    for ascript in scripts:
        if not ascript.getText().startswith('q("talkPage.init"'):
            continue
        # Get the ratings as JSON string
        ratingJSON = re.search('(?<=\"ratings\"\:)(\[.*?\])',ascript.getText()).group(0)
        # Download link
        downlink = str(json.loads(re.search('(?<=\"nativeDownloads\"\:)(\{.*?\})',ascript.getText()).group(0))['low'])
        # Date published
        datepub = datetime.fromtimestamp(int(re.search('(?<=\"published\"\:)(\d*)',ascript.getText()).group(0)))
        # Date Filmed
        datefilm = datetime.fromtimestamp(int(re.search('(?<=\"filmed\"\:)(\d*)',ascript.getText()).group(0)))
        break
    ratings={}
    totcount=0
    for item in json.loads(ratingJSON):
        ratings[str(item['name']).lower()]=item['count']
        totcount+=item['count']
    ratings['total_count']=totcount
    # Date Crawled
    datecrawl = datetime.now()

    return {'ratings':ratings,'title':title,'author':author,'keywords':keywrds,\
    'totalviews':totviews,'downloadlink':downlink,'datepublished':datepub,\
    'datefilmed':datefilm,'datecrawled':datecrawl,'vidlen':vidlen}

def crawl_and_save(csvfilename):
    with open(csvfilename,'r') as f:
        data = f.read().split('\r')
        allids = [aline.split(',')[1] for aline in data[1:] if aline.split(',')[1]]
    i = 0
    count = 0
    skipcount = 0
    while allids:
        count+=1
        i%=len(allids)
        anid = allids[i]

        # Remove the entry and continue if the file exists
        if os.path.isfile(str(anid)+'.pkl'):
            print count,i,'skipping ...',allids[i]
            del allids[i]
            continue

        # Formulate the link
        alink = 'http://www.ted.com/talks/view/id/'+anid
        print count,i,'Extracting:',alink,' ... ',

        try:
            sleep(int(1+1.*random()+skipcount*random()))
            txt,micstime = get_trans_spans(alink)
            sleep(1+skipcount*random())
            meta = get_meta(alink)
        except:
            # If can't extract, just continue
            i+=1
            i%=len(allids)
            print 'Could not crawl. Skipping ...'
            skipcount+=1
            continue

        # If successfully extracted, save the data
        skipcount = 0
        cp.dump({'talk_transcript':txt,'transcript_micsec':micstime,'talk_meta':meta},open(str(anid)+'.pkl','wb'))
        # delete the entry
        del allids[i]
        print 'done'
        

def download_video(url_link):
    pass

if __name__=='__main__':
    csvfilename = './TED Talks as of 01.04.2017.csv'
    crawl_and_save(csvfilename)
