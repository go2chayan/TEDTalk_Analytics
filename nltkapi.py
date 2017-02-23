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
sheetname = 'talk2'
wb = xlrd.open_workbook(bookname)
sh = wb.sheet_by_name(sheetname)
book = copy(wb)
sheet = book.get_sheet(sheetname)


def writefile(i,l,nltkneg,nltkneu,nltkpos):
	sheet.write(i, 0, i)
	#sheet.write(i, 1, str(l))
	sheet.write(i, 2, float(nltkneg))
	sheet.write(i, 3, float(nltkneu))
	sheet.write(i, 4, float(nltkpos))



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
		value = os.popen("curl -d \"text='%s'\" http://text-processing.com/api/sentiment/" % l).read()
		part1,part2 = value.split(', "neutral": ')
		part1,negv = part1.split('"neg": ')
		part2,part3 = part2.split('"pos": ')
		neuv,part2 = part2.split(',')
		posv, part3 = part3.split('}, "label":')
		diff = np.append(diff,float(posv)-float(negv))

		#write to file
		writefile(i,l,negv,neuv,posv)
		i = i+1
		
	book.save(bookname)
	diff=np.convolve(diff,[0.333,0.333,0.333,0.333,0.333])
	j=0
	ii = i
	while j<= (len(diff)-ii): 
		i = i+1
		xval = np.append(xval,i)
		j = j + 1
	ip = np.linspace(1,i,100)
	print i, len(xval), len(diff)
	pl = np.interp(ip,xval,diff)
	return pl
	


fp = open("2.txt")
data = fp.read()
g1 = plotter(data)
plt.plot(g1)
plt.savefig('2.png')


























