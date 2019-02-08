from __future__ import division
import clusteringShots  as cs
import numpy as np
import pysptk
from scipy.io import wavfile
from sys import argv
import wave
import contextlib
import os
import glob
from scipy import signal
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, mixture
from sklearn.metrics import silhouette_samples, silhouette_score
import evaluate_clustering
import evaluation2 as ev2
import evaluate_method
from sklearn.metrics.pairwise import cosine_similarity
import random

AnnotationWeight = 0.0
TranscriptWeight = 0.0
FrequencyWeight = 1
dir_path =""
dirAnnotation =""
dirTranscript =""


def move_files():
	dirAnnotation = dir_path+"annotation"
	dirTranscript = dir_path + "transcript"
	os.mkdir(dirAnnotation)
	os.mkdir(dirTranscript)
	dir_id = 0
	while(os.path.isdir(dir_path+str(dir_id))):

		os.system("cp "+dir_path+str(dir_id)+"/"+"anotation* "+dirAnnotation)
		os.system("cp "+dir_path+str(dir_id)+"/"+"transcript* "+dirTranscript)
		dir_id = dir_id + 1

def get_duration(file):


	with contextlib.closing(wave.open(file,'r')) as f:
	    frames = f.getnframes()
	    rate = f.getframerate()
	    duration = frames / float(rate)
	    return duration

for i in range(66,67):
	dir_path = "/home/eduardo/"+str(i)+"/"
	#move_files()

	num_emphasys = []
	n_chunks = len(glob.glob(dir_path +"chunks/chunk*"))
	freq_matrix = []
	for i in range(0, n_chunks):
		duration = 0
		file =""
		if(i <=9):
			file = dir_path+"chunks/chunk-0"+str(i)+".wav"


		else:
			file = dir_path+"chunks/chunk-"+str(i)+".wav"

		duration = get_duration(file)

		fs, x = wavfile.read(file)
		assert fs == 16000

		f0_swipe = pysptk.swipe(x.astype(np.float64), fs = fs, hopsize = 80, min=60, otype="f0")
		a = []
		f  = []
		X_Frequecies_Vector = []
		for w in f0_swipe:
			if w != 0:
				f.append(w)
		if len(f) >= 30:
			f = random.sample(f, 30)
		else:
			f += [0] * (30 - len(f))
		freq_matrix.append(f)
		#a =  np.var(f)

		#print(len(f))
		#c, Pxx_den = signal.welch(x, fs, nperseg=1024)
		#v = np.var(Pxx_den)
		#if(~np.isnan(v) and ~np.isnan(a)):
			#num_emphasys.append ( v )
		#	l = []
		#	l.append(a)
		#	l.append(v)
		#	freq_matrix[i] = l

	#print(np.all(np.isfinite(freq_matrix)))



	print(freq_matrix)
	afinity_matrix = cosine_similarity(freq_matrix)

	freqMatrix = np.array(afinity_matrix)
	transcript_matrix, annotation_matrix = cs.computeMatrix(dir_path, n_chunks)
	matrixT = np.array(transcript_matrix)
	matrixA = np.array(annotation_matrix)
	#matrix = cs.computeMatrix()
	#afinity_matrix = FrequencyWeight*freqMatrix +  TranscriptWeight*matrixT + AnnotationWeight*matrixA
	np.savetxt("foo.csv", afinity_matrix, delimiter=",")

	min = 5
	max = 40
	print(min)
	exec_times = 10

	nm = 2
	sources_number = 3
	while nm < sources_number:
		h = 0
		best_model_silhouettte = 0
		iterations_without_improvment = 0
		best_model = ''
		if(nm == 0):
			afinity_matrix = freqMatrix*1 + 0*matrixT +0*matrixA
		elif nm == 1:
			afinity_matrix = freqMatrix*0.5 + 0.5*matrixT
		else:
			afinity_matrix = matrixT * 0.333 + 0.333 * matrixA + 0.333 * freqMatrix

		aprox_distance_matrix = 1 - afinity_matrix
		while h < exec_times:
			best_model_silhouettte = -1000
			iterations_without_improvment = 0
			for i2 in range(min, max + 1):
				print(i2)
				model2 = cluster.SpectralClustering(i2, eigen_solver="arpack", affinity='precomputed', n_init=1000, n_jobs=-1, random_state=None)
				cluster_labels = model2.fit_predict(afinity_matrix)
				silhouette_avg = silhouette_score(aprox_distance_matrix, cluster_labels, metric = 'precomputed')
			    	print("For n_clusters =", i2,"The average silhouette_score is :", silhouette_avg)
				if(silhouette_avg > best_model_silhouettte):
					best_model_silhouettte = silhouette_avg
					best_model = model2
					iterations_without_improvment = 0
				else:
					iterations_without_improvment = iterations_without_improvment + 1

				if iterations_without_improvment >= 10:
					break

			for i in range(0, len(best_model.labels_)):
					if(i-1 >= 0 and i+1 < len(best_model.labels_)):
						if ((best_model.labels_[i-1] == best_model.labels_[i+1] and best_model.labels_[i] != best_model.labels_[i-1])
							or best_model.labels_[i-1] != best_model.labels_[i] and best_model.labels_[i] != best_model.labels_[i+1]):
							best_model.labels_[i] = best_model.labels_[i-1]

			u = []
			for i in range(1, len(best_model.labels_)):
				if(best_model.labels_[i] != best_model.labels_[i-1]):
					u.append(i)
			#hg = np.argsort(num_emphasys)
			#u = hg[len(hg)- len(cuts2):len(hg)]
			#print(u)
			solution = []

			#print sorted(u)
			evaluate_method.evaluate(dir_path, u, 'aaa')
			h = h + 1
		nm = nm + 1
