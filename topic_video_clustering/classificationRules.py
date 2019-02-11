


#set of all the grams from the audio modality
gramsA=[]

#set of all the grams form the video 
gramsV=[]

#for every p belonging to GramsA| is selected byNaive Bayes audio classifier}, set of keyphrase based on audio feature
keysA=[]

#for every p belonging to GramsA| is selected by Naive Bayes audio  classifier}, set of keyphrase based on audio feature
keysV=[]

#k,a set with all the selected keyparase alfet applying the rules
k=[]

#if the Phrase Height satisfies the threshold(limiar)
Oh=1

#it is threshold
Oc=1



for p in keysA+keysV:
    k=k+p #Rule 1

for p in keysA+(gramsV-keysV):
    k=k+ { Correlation(p)>0 or PhraseHeight(p)> Oh or Cuewords(p)>Oc)}  }#Rule 2

#keyA + negation gramsV that means (keyA + "all grams" - gramsV)  ???
for p in keysA+gramsV
    k=k+{p or AudioProbability(p)>Ua}#Rule 3

for p in keysA+(gramsA-keysA):
    k=k+ { Correlation(p)>0 or PhraseHeight(p)> Oh or Cuewords(p)>Oc)}  }#Rule 4

#keyA + negation gramsA that means (keyV + "all grams" - gramsA) ???
for p in keysV+GramsA
    k=k+{p or VideoProbability(p)>Uv}#Rule 5
