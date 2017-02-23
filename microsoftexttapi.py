import nltk.data
from string import digits
import fileinput
import re
import nltk.data
from string import digits
import fileinput
import re
import string
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import urllib2
import urllib
import sys
import base64
import json
import csv
import xlrd
import xlwt
from xlutils.copy import copy
import os.path
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

bookname = "Sentimentdata.csv"
sheetname = 'talk1'
wb = xlrd.open_workbook(bookname)
sh = wb.sheet_by_name(sheetname)
book = copy(wb)
sheet = book.get_sheet(sheetname)


def writefile(i,score):
	sheet.write(i, 5, float(score))



def callapi(var):
	base_url = 'https://westus.api.cognitive.microsoft.com/'
	account_key = 'a18f90b602c04d84b90a1057a23b15d0'
	headers = {'Content-Type':'application/json', 'Ocp-Apim-Subscription-Key':account_key}
	num_detect_langs = 1;
	input_texts = '{"documents":[{"id":"1","text":"'+var+'"}]}'
	batch_sentiment_url = base_url + 'text/analytics/v2.0/sentiment'
	req = urllib2.Request(batch_sentiment_url, input_texts, headers) 
	response = urllib2.urlopen(req)
	result = response.read()
	obj = json.loads(result)
	for sentiment_analysis in obj['documents']:
		v = str(sentiment_analysis['score'])
	return sentiment_analysis['score']


def plotter(data):
	pos = 0
	neg = 0
	neu = 0
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	datawithoutnumber = ''.join(i for i in data if not i.isdigit())
	datawithoutbracketval = re.sub("[\(\[].*?[\)\]]", "", datawithoutnumber)
	datawithoutnumcol = datawithoutbracketval.translate(None, ':()\"')
	line = tokenizer.tokenize(datawithoutnumcol)
	i = 0
	xval = np.array([])
	diff = np.array([])
	for l in line:
		if i == 0:
			i = i + 1
			continue
		xval = np.append(xval,i)
		i = i+1
		var = callapi(l)
		diff = np.append(diff,var)
		writefile(i,var)
		i = i+1

	book.save(bookname)
	diff=np.convolve(diff,[0.333,0.333,0.333,0.333,0.333])
	j=0
	ii = i
	while j< (len(diff)-ii): 
		i = i+1
		xval = np.append(xval,i)
		j = j + 1
	ip = np.linspace(1,i,100)
	print i, len(xval), len(diff)
	pl = np.interp(ip,xval,diff)
	return pl
	


fp = open("1.txt")
data = fp.read()
g1 = plotter(data)
plt.plot(g1)
plt.show()
#plt.savefig('2.png')
























