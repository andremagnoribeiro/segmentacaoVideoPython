#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from skm3 import CSOClassifier as CSO
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
# Create an instance of the CSO_classifier class
clf = CSO(version=1)





def annotate(title, abstract):
    input = {"title": title, "abstract" : abstract}
    # Loads CSO data from local file
    clf.load_cso()
    depths = []
    # provides the topics within the paper with an explanation
    result = clf.classify(input, format='json', num_narrower=1, min_similarity=0.9, climb_ont='jfb', verbose=False)
    '''with open('result.json', 'w') as outfile:
        outfile.write(json.dumps(result))'''
    result = list(result)
    auxList = ["http://cso.kmi.open.ac.uk/topics/" + x.replace(" ", "%20") for x in result  ]

    avgDepth = 0
    #result = set(result)
    for item in auxList:
        depth = getDepth(item)
        if depth != -1:
            depths.append(depth)
        else:
            depths.append(10000)
    


    #print(avgDepth)
    return list(result), depths

def getCategory(term):
    categories = []
    sparql = SPARQLWrapper("http://localhost:9999/blazegraph/sparql")
    sparql.setQuery("""select distinct ?x where { ?x <http://www.w3.org/2004/02/skos/core#broaderGeneric> <"""+term+""">}""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        categories.append(str(result["x"]["value"]))
    categories  = [x.split("/")[-1].replace("%20", " ") for x in categories]
    return categories

def getDepth(term):
    categories = []
    sparql = SPARQLWrapper("http://localhost:9999/blazegraph/sparql")
    sparql.setQuery("""select distinct ?x where { ?x <http://www.w3.org/2004/02/skos/core#broaderGeneric> <"""+term+""">}""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    #print(results)
    for result in results["results"]["bindings"]:
        categories.append(str(result["x"]["value"]))
    if(categories):
        return 1 + getDepth(categories[0])
    else:
        if(term != "http://cso.kmi.open.ac.uk/topics/computer%20science"):
            return -1
        else:
            return 1
