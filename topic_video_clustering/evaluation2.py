from __future__ import division
import os
import numpy as np
import csv

def evaluate(dir_path, solution):
	d  = 0
	t = 0
	aligned_segmentation = []
	ground_truth, algorithm_segmentation = get_scenes(dir_path, solution)
	gt = []
	seg = []
	gt.append(0)
	seg.append(0)
	for l in ground_truth:

		gt.append(l[-1])
	for l in algorithm_segmentation:

		seg.append(l[-1])
	gt = set(gt)
	seg = set(seg)
	print(gt, seg)
	
	p = precision(seg,gt)
	r = recall(seg,gt)
	fm = fmeasure(p, r)
	f = open(dir_path+"evaluation.csv", "a")
	fieldnames = ['num_topics_generated', 'precision_mean', 'precision_std', 'precision_median','recall_mean','recall_std', 'recall_median', 'fmeasure_mean','fmeasure_std', 'fmeasure_median'  ]

	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	writer.writerow({'num_topics_generated': len(seg),'precision_mean': np.mean(p), 'precision_std': np.std(p), 'precision_median': np.median(p), 'recall_mean':np.mean(r), 
	'recall_std':np.std(r), 'recall_median': np.median(r), 'fmeasure_mean': np.mean(fm), 'fmeasure_std': np.std(fm), 'fmeasure_median':np.median(fm)})
	
	

	return aligned_segmentation
def intersection(l1, l2):
    lst3 = [value for value in l1 if value in l2]
    return lst3
def precision(seg, gt):
	inter = intersection(seg, gt)
	return float(len(inter)/len(seg))
def recall(seg, gt):
	inter = intersection(seg, gt)
	return float(len(inter)/len(gt))
def fmeasure(p, r):
	return float(2*p*r / p + r)
def get_scenes(dir_path, solution):
	file_id = 1
	dir = 0
	ground_truth = []
	start = 0
	while(os.path.isdir(dir_path+str(dir))):
		scene = []
		scene.append(start)
		while(os.path.exists(dir_path+str(dir)+"/anotation"+str(file_id)+".txt")):
			scene.append(file_id)
			file_id = file_id + 1
		
		start = scene.pop()
		ground_truth.append(scene)
		dir = dir + 1
	ground_truth[len(ground_truth)-1].append(start)
	print(ground_truth)
	
	result = []
	cut = 0
	for i in range(0, len(solution)):
		scene = []
		while(cut < solution[i]):
			scene.append(cut)
			cut = cut + 1

		if(scene):
			result.append(scene)
	scene = []
	for i in range(solution[len(solution) - 1], start + 1):
		scene.append(i)
			
	result.append(scene)

	return ground_truth, result

'''def precision(aligned_segmentation, ground_truth):
	precision = []
	i = 0
	for i in range(0, len(ground_truth)):
		numerator = 0
		denominator = 0
		try:
			for k in aligned_segmentation[i]:
				if k in ground_truth[i]:
					numerator = numerator + 1
				denominator = denominator + 1
			#print(float(numerator/denominator))
			precision.append(float(numerator/denominator))
		except IndexError:
			print("error")
	return precision
	print(np.mean(precision))
	print(np.std(precision))
def recall(aligned_segmentation, ground_truth):
	recall = []
	i = 0
	for i in range(0, len(aligned_segmentation)):
		numerator = 0
		denominator = 0
		try:
			for k in ground_truth[i]:
				if k in aligned_segmentation[i]:
					numerator = numerator + 1
				denominator = denominator + 1
			#print(float(numerator/denominator))
			recall.append(float(numerator/denominator))
		except IndexError:
			recall.append(0.0)
	return recall
	print(np.mean(recall))
	print(np.std(recall))
	

def fmeasure(precision, recall):
	fmeasure = []
	for i in range(len(precision)):
		if(precision[i] != 0 or recall[i] != 0):
			fmeasure.append(2*(precision[i]*recall[i])/(precision[i]+recall[i]))
		else:
			fmeasure.append(0.0)
	return fmeasure
'''
'''def getMetrics(aligned_segmentation, ground_truth):
	f_measure = []
	coverage = []
	precision = []
	i = 0
	for i in range(0, ground_truth):
		for j in aligned_segmentation:
			denominatorP = 0
			denominatorC = 0
			for k in j:'''
				






