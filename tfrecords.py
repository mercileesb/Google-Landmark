import sys
import tensorflow as tf
import os
from utils.utils import *
import pandas as pd
import argparse
import csv

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float32_feature(value):
#	return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in value.tolist()[0]]))
#	print(value.reshape(-1))
	return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def make_traincsvfile():
	_FEATURE_ROOT = "../kaggle"
	_LABEL_LIST = os.listdir(_FEATURE_ROOT)

	
	f = open("train_SENet5_max1000.txt","w")
	
	
	for label in _LABEL_LIST:
		
		featurefolder = os.listdir(os.path.join(_FEATURE_ROOT, label))
		featurefolder = featurefolder[:1000]
		for featurefile in featurefolder:
			
			s = "{}\t{}\n".format(str(featurefile), str(label))
			f.write(s)
	
	f.close()


def make_testcsvfile():
	_FEATURE_ROOT = "../kaggle_test"
	_FEATURE_LIST = os.listdir(_FEATURE_ROOT)

	f = open("test_SENet5_max1000.txt", "w")

	for featurefile in _FEATURE_LIST:
		s = "{}\n".format(str(featurefile))
		f.write(s)

	f.close()
	

def read_csvfile():
	
	pass

def load_feature(filepath):
	return np.load(filepath)
	
def write_feature_tfrecord(idx, dataset, datatype):
	filename = "./tfrecord/{0}_senet_pool5_{1}.tfrecord".format(datatype, idx)
	frame_dir = "."
	writer = tf.python_io.TFRecordWriter(os.path.join(frame_dir, filename))
	for item in dataset:
		filename = item[0].encode()
		label = item[1]
		featurename = item[2]

		
		feature_path = os.path.join("/workspace/kaggle", label, featurename)
		feature = load_feature(feature_path)

		sample = {'video_id': _bytes_feature(filename),
					'labels': _int64_feature(label),
			          'video_feature': _float32_feature(feature),
		          }


		example = tf.train.Example(features=tf.train.Features(feature=sample))


		writer.write(example.SerializeToString())
	writer.close()


def write_feature_test_tfrecord(idx, dataset, datatype):
	filename = "./tfrecord/{0}_senet_pool5_{1}.tfrecord".format(datatype, idx)
	frame_dir = "."
	writer = tf.python_io.TFRecordWriter(os.path.join(frame_dir, filename))
	for item in dataset:
		filename = item[0].encode()
		label = ""
		featurename = item[1]

		feature_path = os.path.join("/workspace/kaggle_test", featurename)
		feature = load_feature(feature_path)

		sample = {'video_id': _bytes_feature(filename),
		          'labels': _int64_feature(label),
		          'video_feature': _float32_feature(feature),
		          }

		example = tf.train.Example(features=tf.train.Features(feature=sample))

		writer.write(example.SerializeToString())
	writer.close()


def make_tfrecord():
	dataset = get_feature_list()
	print(len(dataset))
	datatype = "train"
	chunks_filelist = list(chunks(dataset, 10240))
	for idx, chunk_item in enumerate(chunks_filelist):
		write_feature_tfrecord(idx, chunk_item, datatype)
		print("Generated {}_senet_pool5_{}.tfrecord".format(datatype, idx))


def make_testtfrecord():
	dataset = get_feature_list_test()
	print(len(dataset))
	datatype = "test"
	chunks_filelist = list(chunks(dataset, 10240))
	for idx, chunk_item in enumerate(chunks_filelist):
		write_feature_test_tfrecord(idx, chunk_item, datatype)
		print("Generated {}_senet_pool5_{}.tfrecord".format(datatype, idx))


if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--filetype", help="train / val / test")
	# args = parser.parse_args()
	#make_testcsvfile()
	make_traincsvfile()
	make_tfrecord()
	
	#make_testtfrecord()

	
	
	