import re
import time
import datetime
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

arglist=[]
for i in range(1,len(sys.argv)):
	 arglist.append(sys.argv[i])
nf = int(arglist[0])
wc = int(arglist[1])
nc = int(arglist[2])


te = ts =time.time()
count = 0
stemmer = nltk.PorterStemmer()  
stpwds = nltk.corpus.stopwords.words("english")
ps = PorterStemmer()
# print stpwds
texts=[]
reviewkwds = open('stemingreviews.txt','w')

errors=0
nouns = set()
with open('review text.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        count += 1
        kwds = []
        # if count >10:
        #     break
        if count%1000 == 0:
            te = time.time()
            print count, ':', te-ts
            ts = te
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", row[1])
        # 3. Convert words to lower case and split them
        words = review_text.lower()
        tokens = nltk.word_tokenize(words)
        for t in tokens:
            try:
                st = ps.stem(t).decode('utf-8')
                if st not in stpwds:
                    kwds.append(st)
                    #worddic.add(st)
                    reviewkwds.write(st + ' ')
            except:
                errors += 1
                continue
        texts.append(kwds)
        tags = nltk.pos_tag(kwds)
        for k, pos in tags:
        	if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
        		nouns.add(k.encode('utf-8'))
        reviewkwds.write('\n')
print "errors:", errors
print 'done segmentation:', datetime.datetime.now()

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = nf    # Word vector dimensionality                      
min_word_count = wc   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(texts, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = str(nf)+"features_"+str(wc)+"minwords_10context"
model.save(model_name)


word_vectors = model.wv
featureidx = dict()
nn_vectors = np.array([])
kcnt = -1
#print model.wv
for key in word_vectors.vocab:
    if key in nouns:
        kcnt += 1
        featureidx[kcnt] = key
        if kcnt == 0:
            nn_vectors = word_vectors[key]
        else:
            nn_vectors = np.vstack((nn_vectors,word_vectors[key]))

print kcnt
##clustering words according to word vectors
from sklearn.cluster import KMeans
ncluster = nc
km = KMeans(n_clusters=ncluster, init='k-means++', max_iter=100, n_init=1)

print("Clustering sparse data with %s" % km)
t0 = time.time()
km.fit(nn_vectors)
print("done in %0.3fs" % (time.time() - t0))

wordcluster = open(str(nf)+'features'+str(wc)+'minwordswordcluster'+str(nc)+'.txt','w')
print "Top terms per cluster:"
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(ncluster):
    print "Cluster %d:" % i
    wordcluster.write("Cluster "+str(i))
    for ind in order_centroids[i, :10]:
        print ' %s' % featureidx[ind]
        wordcluster.write(' '+featureidx[ind])
    wordcluster.write('\n')