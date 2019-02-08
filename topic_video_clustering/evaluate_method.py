from __future__ import division

import json
import collections
import re
def find_times(file_path):
	file = open(file_path, 'r')
	f = file.read()
	times = []
	timesEnd = []
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
	return times, timesEnd

def evaluate(dir_path,solution, ground_truth_json_path):
	times, timesEnd = find_times(dir_path+"seg.txt")
	with open('gt_jjvBnvA8GzA.json') as f:
		data = json.load(f)
		ground_truth = sorted(map(int, data.keys()))
		print(ground_truth)
		hits = 0
		for gt in ground_truth:

			for u in solution:
				if (times[u] - 10 <= gt  and times[u] + 10 >= gt ):
					print(gt, times[u], u)
					hits = hits + 1
					break
		print(hits)

		precision  = float(hits/len(solution))
		recall = float(hits/len(ground_truth))
		fscore = 2* float((precision * recall) / (precision + recall))
		print(precision, recall, fscore)
