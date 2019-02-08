import http.client
import urllib
from bs4 import BeautifulSoup
categories = []
supercategories = []
resources = []
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
def sameAs(resource):
	 print("here1")

	 return dbpediaGet("distinct ?x where {?x owl:sameAs <" + resource + ">}")

def getCategory(resource):
	print("here2")
	return dbpediaGet("distinct ?x where {<" + resource + ">  dct:subject ?x}");


def getBroaderCategory(category):
	return dbpediaGet("distinct ?x where {<" + category + ">  skos:broader ?x}")

def dbpediaGet(query):
	l = []
	conn = http.client.HTTPConnection("dbpedia.org:80")
	query= query.replace(" ", "%20")
	#print(query)
	conn.request("GET", "/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=select+"+query+"+LIMIT+5&timeout=300&debug=on")
	response = conn.getresponse().read()

	for item in response.split("</uri>"):
		if "<uri>" in item:
			a = item [ item.find("<uri>")+len("<uri>") : ]
			#print(a)
			l.append(a)


	return l

#resources = sameAs("http://pt.dbpedia.org/resource/Camada_de_transporte")
#print(resources)

def getResourcesAndCategories(resource):
	global resources
	global categories
	global supercategories
	resources.append(resource)

	#for x in resources:
	categories =  getCategory(resource)

	'''for y in categories:
		supercategories = supercategories + getBroaderCategory(y)'''

	#print resources + categories + supercategories
	return resources + categories
