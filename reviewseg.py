# -*- coding: UTF-8 -*-
#import gensim
#from gensim import corpora, models, similarities
import json
import csv
import nltk
from nltk.stem import PorterStemmer
import re
import time
import datetime
 
te = ts =time.time()
count = 0
stemmer = nltk.PorterStemmer()  
stpwds = nltk.corpus.stopwords.words("english")
ps = PorterStemmer()
# print stpwds
r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~Â]+'
texts=[]
reviewkwds = open('reviewkwds.txt','w')


errors=0
with open('review text.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		count += 1
		kwds = []
		# if count >10:
		# 	break
		if count%1000 == 0:
			te = time.time()
			print count, ':', te-ts
			ts = te
		sens=re.split('.!',row[1])
		for s in sens:
			cleanpunc = re.sub(r, "", s)
			cleanreview = re.sub('[0-9]+', '', cleanpunc)
			tokens = nltk.word_tokenize(cleanreview)
			tags = nltk.pos_tag(tokens)
			for k, pos in tags:
				#print type(k)
				# print t
				if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
					try:
						t = ps.stem(k.lower().decode('utf-8'))
						# stem = stemmer.stem(t.lower())
						# if  unicode(stem) not in stpwds:
						if t not in stpwds:
							kwds.append(t.encode('utf-8'))
							reviewkwds.write(t.encode('utf-8')+' ')
					except:
						errors += 1
						continue
			texts.append(kwds)
		reviewkwds.write('\n')

print "errors:", errors
##generate dictionary
print 'done segmentation:', datetime.datetime.now()