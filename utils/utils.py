import os
import numpy as np




# ## test code
# 
# filename = ["ccf9e380ddb25e26.jpg_SENet_pool5.npy", "22780287073e5183.jpg_SENet_pool5.npy"]
# 
# 
# 
# for i in filename:
# 	filepath = "../../kaggle/00598/" + str(i) 
# 	a = np.load(filepath)
# 	
# 	#a = np.fromfile(filepath, dtype="float32")#.reshape(-1, 2048)
# 	print (a.shape)
# 	print (a)


def get_feature_list():
	feature_list = []
	txt_file = "{}_{}.txt".format("train", "SENet5")
	f = open(txt_file, "r")
	while True:
		line = f.readline()
		if not line:
			break
			
		featurename = line.split('\t')[0]
		label = line.split('\t')[1].split('\n')[0]
		filename = featurename.split(".")[0]
		feature_list.append([filename, label, featurename])
		
	return feature_list


def get_feature_list_test():
	feature_list = []
	txt_file = "{}_{}.txt".format("test", "SENet5")
	f = open(txt_file, "r")
	while True:
		line = f.readline()
		if not line:
			break

		
		featurename = line.split('\n')[0]
		
		#label = line.split('\t')[1].split('\n')[0]
		filename = featurename.split(".")[0]
		feature_list.append([filename, featurename])
		
		
	return feature_list


def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]
		


