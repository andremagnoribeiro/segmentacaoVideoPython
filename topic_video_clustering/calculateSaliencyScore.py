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
from scipy import stats

dir_path =""
dirAnnotation =""
dirTranscript =""

lamba = 0.7
gamma = 0.4
alfa = 0.3

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

def getranking(n_chunks, emphasis_vector_list, transcript_distance_matrix, annotation_distance_matrix, N_rank):

	'''list of salience scores for chunk'''
	saliency_score = []

	'''list of emphasis norm for each audio chunk'''
	#emphasis_norm = []
	#for empashis_vector in emphasis_vector_list:
		#emphasis_norm.apdef createDependencyTree():
    pend(np.sqrt(np.power(empashis_vector[0], 2) + np.power(empashis_vector[1], 2)))

	for chunk_index in range(n_chunks):
		max_distance_transcription = 0
		max_distance_annotation = 0

		if(chunk_index == 0):
			max_distance_transcription = transcript_distance_matrix[chunk_index][chunk_index+1]
			max_distance_annotation = annotation_distance_matrix[chunk_index][chunk_index+1]
		elif (chunk_index == n_chunks - 1):
			max_distance_transcription = transcript_distance_matrix[chunk_index][chunk_index-1]
			max_distance_annotation = annotation_distance_matrix[chunk_index][chunk_index-1]
		else:
			#print(chunk_index)
			if(transcript_distance_matrix[chunk_index][chunk_index+1] > transcript_distance_matrix[chunk_index][chunk_index-1]):
				max_distance_transcription = transcript_distance_matrix[chunk_index][chunk_index+1]
			else:
				max_distance_transcription = transcript_distance_matrix[chunk_index][chunk_index-1]

			if(annotation_distance_matrix[chunk_index][chunk_index+1] > annotation_distance_matrix[chunk_index][chunk_index-1]):
				max_distance_annotation = annotation_distance_matrix[chunk_index][chunk_index+1]
			else:
				max_distance_annotation = annotation_distance_matrix[chunk_index][chunk_index-1]

		salience =  (-lamba * max_distance_transcription)  + alfa * max_distance_annotation + 0.001* emphasis_vector_list[chunk_index][0]
		+  0.001* float((1/n_chunks - chunk_index + 1)) * emphasis_vector_list[chunk_index][1]
		print(salience)
		saliency_score.append(salience)

	return sorted(range(len(saliency_score)), key=lambda x: saliency_score[x])[-N_rank:]

def processingVideo():

	for i in range(66,67):
		dir_path = "/home/eduardo/data_base_www/2PhaT6AbH3Q/"
		#move_files()

		num_emphasys = []
		n_chunks = len(glob.glob(dir_path +"chunks/chunk*"))
		freq_matrix = [[0 for i in range(2)] for j in range(n_chunks)]
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

			f0_swipe = pysptk.swipe(x.astype(np.float64), fs = fs, hopsize = 20, min=60)
			a = []
			f  = []
			X_Frequecies_Vector = []
			for w in f0_swipe:
				if w != 0:
					f.append(w)


			#if not f:
			#	a = 0
			#else:
			#	a = stats.mode(f)[0][0]
			a = np.mean(f)



			#print(len(f))
			c, Pxx_den = signal.welch(x, fs, nperseg=1024)
			#if(Pxx_den.any()):
			#	v = 0
			##	v = stats.mode(Pxx_den)[0][0]
			v = np.mean(Pxx_den)
			if(~np.isnan(v) and ~np.isnan(a)):
				#num_emphasys.append ( v )
				l = []
				l.append(a)
				l.append(v)
				freq_matrix[i] = l



		weightA = [0 for i in range(n_chunks)]
		weightT = [0 for i in range(n_chunks)]
		transcript_matrix, annotation_matrix, avg_depth = cs.computeMatrix(dir_path, n_chunks)
		best_model_silhouettte = -1000
		iterations_without_improvment = 0
		max = 32

		model2 = cluster.SpectralClustering(max, affinity='precomputed', n_init=10000, n_jobs=-1)
		cluster_labels = model2.fit_predict(transcript_matrix)



		for i in range(len(cluster_labels)-1):
			if(cluster_labels[i] != cluster_labels[i+1]):
				weightT[i+1] = np.sqrt(pow(freq_matrix[i+1][0],2) + pow(freq_matrix[i+1][1],2))

		for j in range(len(annotation_matrix) -1):
			if(not set(annotation_matrix[j]).intersection(annotation_matrix[j+1])):
				weightA[j+1] =  float(np.sqrt(pow(freq_matrix[j+1][0],2) + pow(freq_matrix[j+1][1],2)) /abs(avg_depth[j] - avg_depth[j+1]))

		#rankingT = sorted(range(len(weightT)), key=lambda x: weightT[x])[-70:]
		rankingA = sorted(range(len(weightA)), key=lambda x: weightA[x])[-70:]
		#ranking = list(set(rankingT).intersection(rankingA))
		#merged = sorted(list(set(ranking).union(rankingA)))
		#matrixT = np.array(transcript_matrix)
		#matrixA = np.array(annotation_matrix)
		#ranking = getranking(n_chunks, freq_matrix, matrixT, matrixA, 25)
		evaluate_method.evaluate(dir_path, sorted(rankingA), "aas")

processingVideo()
