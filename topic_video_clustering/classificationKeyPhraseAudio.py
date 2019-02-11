import numpy as np
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk import FreqDist
import operator
import nltk
from math import log2



def readingTranscription():
    y=0
    quantTranscription=20;
    video_path="../jjvBnvA8GzA/"
    numberWords=0
    print("\n aquivos lidos\n")
    listAllWords=[]
    while y < quantTranscription:
        i=0
        f2 = open(video_path + "transcript/transcript"+str(y)+".txt")
        textAux  = f2.read()
        print(textAux)
        text=textAux.split("\n")
        for i in text:
            if(i!=""):
                listAllWords.append(i)
                numberWords+=1
        y+=1
    allWords=""
    for i in listAllWords:
        allWords=allWords+i+" "
    return allWords;


def RemoveStopWords(instance):
    instance = instance.lower()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    palavras = [i for i in instance.split() if not i in stopwords]
    return (" ".join(palavras))



allWordsGlobal=readingTranscription()
allWordsWithoutStopWords =""
allWordsWithoutStopWords = RemoveStopWords(allWordsGlobal)


listAllWords=allWordsWithoutStopWords.split(" ")

k=2
windows=[]

j = 0
count = 0
while j < len(listAllWords):
    window = ''
    for i in range(k):
        if(j+i < len(listAllWords)):
            window = window + listAllWords[j+i] + " "
    
    j = j + k - 1 
     
    windows.append(window)

print("\n\n",windows)


pharaseP=[]

w=len(windows)
for p in windows:

    μp=-1
    σp2=-1

    sum1=0
    for j in windows:
        if(p==j):
            sum1+=1 

    μp= sum1/w

    sum2=0
    for m in window:
        if(p==j):
            sum2+=(1 - μp) ** 2
        else:
            sum2+=(0 - μp) ** 2
            

    σp2=sum2/w

    ATDp=μp/(μp+(σp2 ** 2))

    pharaseP.append([[p]+[μp]+[σp2]+[ATDp]])
    

for i in pharaseP:
    print(i)

    

