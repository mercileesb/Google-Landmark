import csv
import os
import shutil

_PATH = "/media/mmlab/hdd/Shin/recognition/train/" 

f = open('data.csv', 'r')
rdr = csv.reader(f)
for line in rdr:
    folder_name = "{0:05d}".format(int(line[0]))
    src = os.path.join(_PATH, folder_name)
    dst = os.path.join("./smalldataset", folder_name)
    shutil.copytree(src, dst)
     
    
    
f.close()

