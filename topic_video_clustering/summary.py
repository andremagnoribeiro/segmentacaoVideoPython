from data_structures import Shot, AgglomeratedTree
import numpy as np
import pysptk
from scipy.io import wavfile
from scipy.io.wavfile import read
import re
from sys import argv
import wave
import contextlib
import os
import glob
from scipy import signal
import evaluate_method
from joblib import Parallel, delayed
import multiprocessing
import time

class Summary:
    def __init__(self, video_path):
        self.video_path = video_path
        self.chunks_path = self.video_path + "chunks/"
        self.n_chunks = len(glob.glob(self.chunks_path+ "chunk*"))
        self.chunks = []

    '''extract pause duration before being voiced of every audio chunk'''
    def extractPauseDuration(self):
        file_path = self.video_path + "seg.txt"
        file = open(file_path, 'r')
        f = file.read()
        times = []
        timesEnd = []
        pause_list = []
        l = re.findall("\+\(\d*\.\d*\)",f )
        for i in l:
            i = i.replace("+","")
            i = i.replace("(","")
            i = i.replace(")","")
            times.append(float(i))

        l = re.findall("\-\(\d*\.\d*\)",f )
        for i in l:
    	    i = i.replace("-","")
    	    i = i.replace("(","")
    	    i = i.replace(")","")
    	    timesEnd.append(float(i))
        file.close()
        pause_list.append(times[0])
        for i in range(1, len(timesEnd)):
            pause_list.append(float(times[i] - timesEnd[i-1]))

        return pause_list

    '''Extract the pitch and volume estimation of each audio chunk'''
    def extract_emphasis(self, index_chunk):

        chunks_path = self.video_path + "chunks/"

        if(index_chunk <= 9):
            file = self.chunks_path+"chunk-0"+str(index_chunk)+".wav"
        else:
            file = self.chunks_path+"chunk-"+str(index_chunk)+".wav"

        fs, x = wavfile.read(file)
        assert fs == 16000

        f0_swipe = pysptk.rapt(x.astype(np.float32), fs = fs, hopsize = 20, min=60, otype="pitch")
        a = []
        f  = []
        X_Frequecies_Vector = []
        for w in f0_swipe:
            if w != 0:
                f.append(w)

        pitch = float(np.median(f))
        if (np.isnan(pitch)):
            pitch = 0

        #dbs = 20*np.log10( np.sqrt(np.mean(x**2)) )
        dbs = 2000*np.log10(np.sqrt(np.mean(x**2))) / 5*(self.n_chunks - index_chunk)

        if(np.isnan(dbs)):
            dbs = 0


        return pitch, dbs


    '''Method that create a audio chunk object'''
    def createShots(self, i, pause, ocr_on):
        pitch, volume = self.extract_emphasis(i)
        s = Shot(i, pitch, volume, pause)
        s.extractTranscriptAndConcepts(summary.video_path, ocr_on)
        s.buildDependencyTree()
        return s

        #MOV(p)= Cp(is total number of occurrences of p in all the slides)/
        #|S|(total number of slides.)
    #def meanOccurrenceRatio():

    #def contiguousOccurrenceRatio():

    #def phraseHeight():
    #def naiveBayes();


if __name__ == '__main__':
    start_time = time.time()

    summary = Summary(argv[1])
    ocr_on = False
    try:
        ocr_on = argv[2]
        if ocr_on == 'ocr':
            ocr_on = True
    except IndexError:
        ocr_on = False
    num_cores = multiprocessing.cpu_count()
    pauses = summary.extractPauseDuration()
    shots = []
    chunks = Parallel(n_jobs=num_cores)(delayed(summary.createShots)(i, pauses[i], ocr_on) for i in range(summary.n_chunks))
    summary.chunks = chunks


    agt = AgglomeratedTree(summary.chunks, ocr_on)
    boundaries = agt.agglomerateT()
    evaluate_method.evaluate(summary.video_path, boundaries, 'aaaa')
    print("--- %s seconds ---" % (time.time() - start_time))
