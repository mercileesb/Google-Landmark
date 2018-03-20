import tensorflow as tf
import os
import numpy as np


input_file1 = os.path.join("/workspace/landmark/tfrecord/", "train_senet_pool5_1.tfrecord")


for serialized_example in tf.python_io.tf_record_iterator(input_file1):
    example = tf.train.SequenceExample()
    example.ParseFromString(serialized_example)

    feature = example.context.feature["video_id"].bytes_list.value
    label = example.context.feature["labels"].int64_list.value
    #rgb = example.feature_lists.feature_list["audio"].feature
    #rgb = example.feature_lists.feature_list["rgb"].feature

    rgb = example.context.feature["video_feature"].float_list.value

    #print len(rgb[:])
    print("Feature: {}, label: {}, Len: {}".format(feature, label, len(rgb)))
    #print(rgb)
    #print ("{}   {}     {}".format(feature, label, len(rgb)))
    
    
    # a = rgb[0].bytes_list.value[0]  # [0].decode('utf-8')
    # print (np.fromstring(a))
    # print (len(a))
    # 
    # b = np.fromstring(a, dtype=np.float32)
    # print (b)
    # print (len(b))


    break

# 
# for serialized_example in tf.python_io.tf_record_iterator(input_file1):
#     example = tf.train.SequenceExample()
#     example.ParseFromString(serialized_example)
# 
#     feature = example.context.feature["video_id"].bytes_list.value
#     label = example.context.feature["labels"].int64_list.value
#     #rgb = example.feature_lists.feature_list["audio"].feature
#     rgb = example.feature_lists.feature_list["rgb"].feature
#     #rgb = example.feature_lists.feature_list["frame_feature"].feature
# 
#     #print len(rgb[:])
#     #print("Feature: {}, label: {}, Len: {}".format(feature, label, len(rgb)))
# 
#     print len(rgb)
#     a = rgb[0].bytes_list.value[0]  # [0].decode('utf-8')
#     print len(a)
# 
#     break

