from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster, datasets, mixture
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_distances
import annotate
import sys
sys.path.insert(0, 'document_similarity/')
from document_similarity import DocSim
from gensim.models.keyedvectors import KeyedVectors
def computeMatrix(dir_path, num_chunks):
	#stopwords = nltk.corpus.stopwords.words('english')

	'''features from annotation'''
	'''i = 0
	docs = []
	for i in range(0, num_chunks):
		aux =""
		auxList = []
		f = open(dir_path+"annotation/anotation"+str(i)+".txt")
		a  = f.readlines()
		for y in a:
			if(y != "--"):
				#print(y)
				auxList = auxList + dbpedia.getResourcesAndCategories(y.replace("\n",""))
				#print(auxList)
			else:
				auxList.append(y)
		for x in auxList:
			x = x.replace("http://pt.dbpedia.org/resource/","")
			x = x.replace("http://dbpedia.org/resource/", "")
			x = x.replace("http://dbpedia.org/resource/Category:","")
			x = x.replace("_","")
			x = x.replace(" ","")
			aux =  aux +" "+ x
		aux = aux.replace("\n","")
		docs.append(aux)
		#print(docs)
		#print("aaa")
	vectorizer = TfidfVectorizer(encoding='utf-8',sublinear_tf=True)
	X = vectorizer.fit_transform(docs)
	print('saiu')
	afinity_matrix2 = cosine_distances(X)
'''



	# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
	# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
	# Feel free to use to your own model.
	googlenews_model_path = 'document_similarity/data/GoogleNews-vectors-negative300.bin'
	stopwords_path = "document_similarity/data/stopwords_en.txt"
	stopwords = []
	model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
	with open(stopwords_path, 'r') as fh:
	    stopwords = fh.read().split(",")
	ds = DocSim.DocSim(model,stopwords=stopwords)

	'''features from transcription'''
	docsA = []
	docs2 = []
	simMatrixT = []
	avg_depths = []
	previousAnnotation = ['empty']
	previousDepth = [1000]
	stemmer = PorterStemmer()
	docsT = []
	for i in range(0, num_chunks):
		aux = ""
		f2 = open(dir_path+"transcript/transcript"+str(i)+".txt")
		a  = f2.read()

		words = tokenize.word_tokenize(a, language='english')
		words=[word.lower() for word in words if word.isalpha() ]
		preAnnotateText = ' '.join(words)
		if(not preAnnotateText):
			preAnnotateText = 'chemestry dog wolf bug'
		docsT.append(preAnnotateText)
		annotatedTerms, depth = annotate.annotate(preAnnotateText)
		if not annotatedTerms:
			annotatedTerms = previousAnnotation
			depth = previousDepth

		else:
			previousAnnotation = annotatedTerms
			previousDepth = depth
		print(annotatedTerms)
		docsA.append(annotatedTerms)
		avg_depths.append(depth)
	for i in range(0, num_chunks):
		source_doc = docsT[i]
		target_docs = []
		auxSimMT = []
		for j in range(0, num_chunks):
			target_docs.append(docsT[j])

		sim_scores = ds.calculate_similarity(source_doc, target_docs)
		for sim in sim_scores:
			auxSimMT.append(float(sim['score']))
		#print(len(auxSimMT))
		simMatrixT.append(auxSimMT)

		#words=[stemmer.stem(word) for word in words ]
		#aux = ' '.join(words)

		#docs2.append(aux)
	#source_doc = "how to delete an invoice"
	#target_docs = ['delete a invoice', 'how do i remove an invoice', "purge an invoice"]




	#vectorizer2 = TfidfVectorizer(stop_words=stopwords, encoding='utf-8',sublinear_tf=True)
	#X2 = vectorizer2.fit_transform(docs2)

	'''distance_matrix_A = []
	auxVector = []
	for i in range(0, num_chunks):
		for j in range(0, num_chunks):
			intersection_size = len(set(docsA[i]).intersection(docsA[j]))
			auxVector.append(float(intersection_size/len(docsA[i])))
		distance_matrix_A.append(auxVector)'''


	#afinity_matrix = cosine_distances(X2)

	#print(distance_matrix_A)
	#print(simMatrixT)
	return simMatrixT, docsA, avg_depths
