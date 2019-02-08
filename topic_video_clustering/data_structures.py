import networkx as nx
import annotate
from nltk import tokenize
import matplotlib.pyplot as plt
from collections import Mapping
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import sys
import heapq
sys.path.insert(0, 'document_similarity/')
from document_similarity import DocSim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, Birch
from  sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score
from statsmodels import robust

import random
'''Shot representation'''
class Shot:
    def __init__(self, id, pitch, volume, pause):
        self.id = id            #shot id
        self.pitch = pitch      #pitch value
        self.volume = volume    #volume contained in a chunk
        self.pause_duration  = pause # pause time before the shot being voiced
        self.transcript = None  #transcription from ASR of a shot
        self.ocr = None #text extracted from ocr
        self.conceptNodes = [] #list of concept nodes
        self.tree = None  #dictionary representation
        self.root = None  #root of tree representation of a shot

    '''extract the transcripts and related concepts from CSO ontology'''
    def extractTranscriptAndConcepts(self, video_path, ocr_on):

        aux = ""
        f2 = open(video_path + "transcript/transcript"+str(self.id)+".txt")
        a  = f2.read()

        words = tokenize.word_tokenize(a, language='english')
        words=[word.lower() for word in words if word.isalpha() ]
        transcript = ' '.join(words)
        if not transcript:
            transcript = ''

        self.transcript = transcript
        f2.close()

        if(ocr_on):
            aux = ""
            f2 = open(video_path + "slides/frameb"+str(self.id)+".txt")
            a  = f2.read()

            words = tokenize.word_tokenize(a, language='english')
            words=[word.lower() for word in words if word.isalpha() ]
            ocr = ' '.join(words)
            if not ocr:
                ocr = ''

            self.ocr = ocr
            f2.close()
        annotatedTerms = None
        if(ocr_on):
            annotatedTerms, depth = annotate.annotate(self.transcript,  self.ocr)
        else:
            annotatedTerms, depth = annotate.annotate(self.transcript,  self.transcript)


        conceptNodeList = []
        for j in range(len(annotatedTerms)):
            conceptNodeList.append(Concept_Node(j, annotatedTerms[j], depth[j]))
        self.stopwords = []

        self.conceptNodes = conceptNodeList



    '''Build a dependency tree with the concetps nodes'''
    def buildDependencyTree(self):
        G = nx.Graph()
        edges = []
        for i in range(len(self.conceptNodes)):
            for j in range (len(self.conceptNodes)):
                if i != j:
                    edges.append((self.conceptNodes[i].name, self.conceptNodes[j].name, abs(self.conceptNodes[i].depth - self.conceptNodes[j].depth)))
        if edges:
            G.add_weighted_edges_from(edges)

            T = nx.minimum_spanning_tree(G)
            self.root = self.findRoot()
            self.tree = T

        else:
            self.root = self.findRoot()
            T = nx.Graph()
            T.add_node(self.root.name)
            self.tree = T
    '''Find the concept node with the best value of utility function and make it root. In this case, the inverse of the node depth'''
    def findRoot(self):
        lowerDepth  = 100000
        lessDeepNode = None

        for node in self.conceptNodes:
            if node.depth < lowerDepth:
                lowerDepth = node.depth
                lessDeepNode = node

        if(not lessDeepNode):

            lessDeepNode = Concept_Node(-1, 'Dummy Root', 1000)
        return lessDeepNode


'''Nodes of concepts from CSO ontology'''
class Concept_Node:
    def __init__(self, id, name, depth):
        self.id = id
        self.name = name
        self.depth = depth


'''Representation of a agglomerated Node'''

class AgglomeratedNode:
    def __init__(self, val, ids, transcript, ocr, pitch, volume, pause, root, tree):
        self.val = val
        self.transcript = transcript
        self.ocr = ocr
        self.ids = ids
        self.pitch = pitch
        self.volume = volume
        self.pause = pause
        self.root = root
        self.tree = tree
    def __lt__(self, other):
        return self.val < other.val



'''Representation of a tree of agglomerated nodes, which are nodes that represent the union of adjacent shots'''
class AgglomeratedTree:

    def initializeWord2VecModel(self):
        #self.model = KeyedVectors.load_word2vec_format(self.googlenews_model_path, binary=True)
        auxList = []
        auxListPause = []
        auxListVolume = []
        auxListDepth = []
        for s in self.shots:
            auxList.append(s.pitch)
            auxListPause.append(s.pause_duration)
            auxListVolume.append(s.volume)
            auxListDepth.append(s.root.depth)

        self.pitch_mean = np.median(auxList)
        self.std_pitch = robust.mad(auxList)
        self.pause_mean = np.mean(auxListPause)
        self.pause_std = np.std(auxListPause)
        self.volume_mean = np.median(auxListVolume)
        self.volume_std = robust.mad(auxListVolume)
        self.depth_mean = np.median(auxListDepth)
        self.depth_std = robust.mad(auxListDepth)

        print(self.pitch_mean, self.std_pitch)
        print(self.volume_mean, self.volume_std)
        print(self.pause_mean, self.pause_std)
        print(self.depth_mean, self.depth_std)





    def __init__(self, shots, ocr_on):
        self.shots = shots
        self.agglomerate_shots = []
        self.googlenews_model_path = 'document_similarity/data/GoogleNews-vectors-negative300.bin'
        self.stopwords_path = "document_similarity/data/stopwords_en.txt"
        self.stopwords = []
        self.model = None
        self.docSim = None
        self.pitch_mean = 0
        self.std_pitch = 0
        self.pause_mean = 0
        self.pause_std  = 0
        self.volume_mean = 0
        self.volume_std = 0
        self.depth_mean = 0
        self.depth_std = 0
        self.boundaries = []
        self.ocr_on = ocr_on
        self.initializeWord2VecModel()



    def agglomerateT(self):
        j = 0
        for i in range(len(self.shots)):
            self.agglomerate_shots.append(AgglomeratedNode(0, [self.shots[i].id], self.shots[i].transcript, self.shots[i].ocr, self.shots[i].pitch, self.shots[i].volume, self.shots[i].pause_duration, self.shots[i].root, self.shots[i].tree))

        while j <  len(self.shots) - 1:
            for s_index in range(len(self.agglomerate_shots) -1):
                self.agglomerate_shots, has_joined = self.treeUnion(self.agglomerate_shots[s_index], self.agglomerate_shots[s_index + 1])
                if has_joined:
                    break
            j = j + 1
        last_attempt =  []
        boundA = [0]

        if(self.ocr_on):
            for s in self.agglomerate_shots:
                boundA.append(s.ids[0])


            last_attempt = [0]
            for indexes in boundA:
                if (self.shots[indexes].volume > self.volume_mean)  and self.shots[indexes].pause_duration >  self.pause_mean +self.pause_std:

                    last_attempt.append(indexes)
        else:

            for s in self.agglomerate_shots:
                boundA.append(s.ids)


            last_attempt = [0]

            for indexes in range(len(self.shots)):
                if indexes not in boundA and (self.shots[indexes].pitch > self.pitch_mean or self.shots[indexes].volume + self.volume_std  > self.volume_mean )  and self.shots[indexes].pause_duration >  self.pause_mean +self.pause_std:
                    last_attempt.append(indexes)



        print(last_attempt)
        return sorted(list(set(last_attempt)))

    def agglomerate(self, L):

        j = 0

        while j <  len(self.shots) - 1:
            h = []

            for i in range(len(self.agglomerate_shots)):
                if(i == 0):
                    ids = self.agglomerate_shots[i].ids + self.agglomerate_shots[i+1].ids
                    pitch = float((self.agglomerate_shots[i].pitch + self.agglomerate_shots[i+1].pitch) / 2)
                    transcripts = self.agglomerate_shots[i].transcript + ' ' + self.agglomerate_shots[i+1].transcript
                    aNode = AgglomeratedNode(self.textualDistance(self.agglomerate_shots[i], self.agglomerate_shots[i+1]),
                    ids, transcripts, pitch, self.agglomerate_shots[i].root,self.agglomerate_shots[i].tree)
                    heapq.heappush(h, aNode)
                elif(i == len(self.agglomerate_shots) - 1):
                    pitch = float((self.agglomerate_shots[i].pitch + self.agglomerate_shots[i-1].pitch) / 2)
                    ids = self.agglomerate_shots[i].ids + self.agglomerate_shots[i-1].ids
                    transcripts = self.agglomerate_shots[i].transcript + ' ' + self.agglomerate_shots[i-1].transcript

                    aNode = AgglomeratedNode(self.textualDistance(self.agglomerate_shots[i], self.agglomerate_shots[i-1]),
                     ids, transcripts, pitch, self.agglomerate_shots[i].root, self.agglomerate_shots[i].tree )
                    heapq.heappush(h, aNode)
                else:
                    pitch = float((self.agglomerate_shots[i].pitch + self.agglomerate_shots[i+1].pitch) / 2)

                    ids = self.agglomerate_shots[i].ids + self.agglomerate_shots[i+1].ids
                    transcripts = self.agglomerate_shots[i].transcript + ' ' + self.agglomerate_shots[i+1].transcript
                    aNode = AgglomeratedNode(self.textualDistance(self.agglomerate_shots[i], self.agglomerate_shots[i+1]),
                     ids, transcripts, pitch, self.agglomerate_shots[i].root, self.agglomerate_shots[i].tree)
                    heapq.heappush(h, aNode)

                    pitch = float((self.agglomerate_shots[i].pitch + self.agglomerate_shots[i-1].pitch) / 2)

                    ids = self.agglomerate_shots[i].ids + self.agglomerate_shots[i-1].ids
                    transcripts = self.agglomerate_shots[i].transcript + ' ' + self.agglomerate_shots[i-1].transcript

                    aNode2 = AgglomeratedNode(self.textualDistance(self.agglomerate_shots[i], self.agglomerate_shots[i-1]),
                     ids, transcripts, pitch, self.agglomerate_shots[i].root, self.agglomerate_shots[i].tree)
                    heapq.heappush(h, aNode2)

            node = heapq.heappop(h)

            print(node.ids)

            for id in node.ids:
                self.agglomerate_shots = [s for s in self.agglomerate_shots if id not in s.ids]
            self.agglomerate_shots.append(node)
            self.agglomerate_shots = sorted(self.agglomerate_shots, key = lambda x : x.ids)
            j = j + 1

            if(len(self.agglomerate_shots) == L):
                break

        boundaries = []
        for node in self.agglomerate_shots:
            boundaries.append(node.ids[0])

        return boundaries




    def textualDistance(self, shot1, shot2):
        return float( abs(shot2.root.depth - shot1.root.depth) ) / shot1.pitch * (1 + float(self.docSim.calculate_similarity(shot1.transcript, shot2.transcript)[0]['score']))



    def treeUnion(self, agglomeratedShot1, agglomeratedShot2):
        has_joined = False
        subtree_at_2 = None
        tree1 = None
        if (agglomeratedShot1.root.name in nx.to_dict_of_dicts(agglomeratedShot2.tree) and (agglomeratedShot1.pause > agglomeratedShot2.pause)):
            subtree_at_2 = dfs_tree(agglomeratedShot2.tree, agglomeratedShot1.root.name)
            children = nx.to_dict_of_dicts(subtree_at_2)
            tree1 = nx.to_dict_of_dicts(agglomeratedShot1.tree)
            tree1[agglomeratedShot1.root.name] = children
            tree1 = nx.from_dict_of_dicts(tree1)

            has_joined = True
        else:

            for key in nx.to_dict_of_dicts(agglomeratedShot1.tree)[agglomeratedShot1.root.name].keys():
                if key in nx.to_dict_of_dicts(agglomeratedShot2.tree):
                        subtree_at_2 = dfs_tree(agglomeratedShot2.tree, key)
                        children = nx.to_dict_of_dicts(subtree_at_2)
                        tree1 = nx.to_dict_of_dicts(agglomeratedShot1.tree)
                        tree1[key] = children
                        tree1 = nx.from_dict_of_dicts(tree1)

                        has_joined = True



        if(has_joined):
            ocr = ''
            ids = agglomeratedShot1.ids + agglomeratedShot2.ids
            transcript =  agglomeratedShot1.transcript + ' ' + agglomeratedShot2.transcript
            if self.ocr_on:
                ocr = agglomeratedShot1.ocr + ' ' + agglomeratedShot2.ocr

            pitch = agglomeratedShot1.pitch
            volume = agglomeratedShot1.volume
            pause = agglomeratedShot1.pause
            root  = agglomeratedShot1.root
            join = agglomeratedShot1.ids + agglomeratedShot2.ids
            self.boundaries.append(join[0])
            T = nx.minimum_spanning_tree(tree1)

            a_node = AgglomeratedNode(0, ids , transcript, ocr, pitch, volume, pause, root, T)
            for id in a_node.ids:
                self.agglomerate_shots = [s for s in self.agglomerate_shots if id not in s.ids]
            self.agglomerate_shots.append(a_node)
            self.agglomerate_shots = sorted(self.agglomerate_shots, key = lambda x : x.ids)

        return self.agglomerate_shots, has_joined
