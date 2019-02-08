import numpy as np
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk import FreqDist
import operator
import nltk

from math import log2


import re

class Phrase:
    def __init__(self):
        self.phrase=""
        self.codPhrase=-1
        self.codShot=-1
        self.shotInitial=-1
        self.shotEnd=-1


        #frequency no total 
        self.frequency=-1
        #frequency continua 
        self.frequencySequence=-1
        #phrase no topo da pagina
        self.ehPhrasesTitle=0


        self.meanOccurrence=-1 
        self.ContiguousOccurrence=-1
        self.meanHeightPhrase=-1
        
       
    



class FunctionsPhrase:
    def __init__(self):
        
        self.key_phrases = []
        self.all_phrases = []
        self.dict_phrases = []
        self.phrasesTopPage=[]
      


    def leituraArquivosSlides(self,video_path,quantSlide):
        aux = ""
        y=0
        self.key_phrases = []
        listPhrases=[]

        #while para por cada aquivo txt do slide   
        while y < quantSlide:


            i=0
            f2 = open(video_path + "slides/framem"+str(y)+".txt")
            textAux  = f2.read()
            text=textAux.split("\n\n")
            

            # while passa por cada phrase do text separada por \n\n(paragrafo)
            while i < len(text):
                phrase=self.RemoviCaracteres(text[i])
                phrase= phrase.replace("\n"," ")


                if self.PhraseVasia(phrase):
                    f=-1
                    f=self.existsInList2(phrase)
                    
                    linha=[]
                    linha=[phrase]+[str(y)]
                    self.all_phrases=self.all_phrases+[linha]
                    if f<90:
                        self.key_phrases= self.key_phrases+[phrase]
                        
                        
                        #criar um objeto da classe Phrase e acrescenta na lista 
                        p=Phrase()
                        p.codShot=y
                        if i==0:
                            p.ehPhrasesTitle=1
                        p.phrase=phrase
                        listPhrases.append(p)
                            
                    
                
                i=i+1

            y=y+1
        return listPhrases;
    
            
    #retira espaçoes em branco que se formaram na hora do split("\n\n")
    def PhraseVasia(self,phrase):
        for i in range(len(phrase))  :
            if phrase[i]!=" ":
                return True
        return False
   #verifica se existe a phrase na lista
    def existsInList2(self,string):
        i=0
        f=-1
        while i < len(self.key_phrases):
            f=fuzz.ratio(string,self.key_phrases[i])
            aux=0
            if f>90:
                return f
            i=i+1
            if aux<f:
                aux=f
        return f
    #remove os caracteres indesejados
    def RemoviCaracteres(self,instancia):
        
        instancia = instancia.lower()
        palavras=[]
        instancia2=re.sub(r'[-°—?,~./!,":;()]','',instancia)
        palavras = [i for i in instancia2.split() if i.isalpha()]
        return self.RemoviStopWords(" ".join(palavras))

    #remove StopWords
    def RemoviStopWords(self,instancia):
        instancia = instancia.lower()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        palavras = [i for i in instancia.split() if not i in stopwords]
        return (" ".join(palavras))

    # acrescenta a frequency 
    def setfrequency(self,listPhrases):
        phrase=[]
        freq=[]
        matriz=[]
        

        for j in listPhrases:
            phrase=j.phrase

            cont=0
            linha=[]
            linha.append(phrase)

            frequencyNaSequencia=[]
            shotInitial=-1
            shotEnd=-1

           
            for i in range(0,len(self.all_phrases)):
                x=False

                f=0
             
                f=fuzz.ratio(phrase,self.all_phrases[i][0])
             

                if f>90:
                    
                    
                    frequencyNaSequencia.append(self.all_phrases[i][1])
                    
                    

                    cont+=1
        
            j.frequency=cont

            #acha a frequency seguencia das palavras
            g=self.frequencySeguencia(frequencyNaSequencia)
            if g!=[]:

                j.shotInitial=int(g[0])
                j.shotEnd=int(g[len(g)-1])
                j.frequencySequence=j.shotEnd-j.shotInitial
            else:
                j.shotInitial=0
                j.shotEnd=0
                j.frequencySequence=0


        return listPhrases

    #acha a sequencia de shot correspondende a phrase 
    def frequencySeguencia(self,lista):
        n=lista

        g=[]
        x=True
        for i in range(0,len(n)-1):
   
            if int(n[i])+1==int(n[i+1]) or int(n[i])+2==int(n[i+1]) or int(n[i])==int(n[i+1])  or int(n[i])==int(n[i+1]) :
                g.append(n[i])
                if i==len(n)-2:
                    g.append(n[i+1])
                
            else:
                g=[]
            
        return g
           

                

    #ordena matriz pela frequency
    def ordenaMatrizfrequency(self,listPhrases):
        listPhrases.sort(key=lambda x: x.frequency, reverse=True)
        return listPhrases
    
    def ordenaMatrizfrequencySequence(self,listPhrases):
        listPhrases.sort(key=lambda x: x.frequencySequence, reverse=True)
        return listPhrases

    def procedure(self,quantSlide):

        listPhrases=self.leituraArquivosSlides("../jjvBnvA8GzA/",quantSlide)
        listPhrases=self.setfrequency(listPhrases)
        listPhrases=self.ordenaMatrizfrequencySequence(listPhrases)
        listPhrases=self.ordenaMatrizfrequency(listPhrases)

        return listPhrases



quantSlide=154
p=FunctionsPhrase()
listPhrase=p.procedure(quantSlide)

#-------------------K-MEANS-------------------------



listaKmeans=[]

for i in listPhrase:
  
    #print(i.codShot,"f->" ,i.frequency, " fs->",i.frequencySequence,i.ehPhrasesTitle,"(",i.shotInitial,":",i.shotEnd,")",i.phrase)
    


    #Mean occurrence ratio
    i.meanOccurrence=i.frequency/quantSlide

    #Contiguous occurrence ratio    CORV (p) = CC(p)/|S|  × log2T
    i.ContiguousOccurrence=(i.frequencySequence/quantSlide)*log2(i.frequency)

    #Mean occurrence ratio MORV (p) = Cp/|S|
    i.ehPhrasesTitle=i.ehPhrasesTitle
    
    #Points that Kmeans use for clustering
    listaKmeans.append([i.meanOccurrence, i.ContiguousOccurrence])

pontosKmeans = np.array(listaKmeans) 



def clustering_K_means(pontos):
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn import cluster



    y_kmeans=[]
    #print(len(pontos),len(y),type(pontos),type(y_kmeans))
   
    
    #ira agrupar sem 2 grupos um será o grupo de palavras chaves e o outro será o grupo de não palavras chaves
    kmeans = cluster.MiniBatchKMeans(n_clusters=2, batch_size=10)
    
    y_kmeans = kmeans.fit_predict(pontos)


    for i in range(0,len(pontos)):
        
        if y_kmeans[i]==0:
            print('\033[31m'+'0'+'\033[0;0m',pontos[i],y_kmeans[i],listPhrase[i].phrase,"\n")
    for i in range(0,len(pontos)):
        if y_kmeans[i]==1:
            print('\033[32m'+'1'+'\033[0;0m',pontos[i],y_kmeans[i],listPhrase[i].phrase,"\n")
        
        
    # desenha os pontos no gr ́afico
    # as cores s~ao definidas pelo valor de y (grupo) e
    # h ́a contorno nos c ́ırculos (edgecolor)
    plt.scatter(pontos[:, 0], pontos[:, 1], marker='o', c=y_kmeans, s=25, edgecolor='k')
    plt.show()


clustering_K_means(pontosKmeans)










