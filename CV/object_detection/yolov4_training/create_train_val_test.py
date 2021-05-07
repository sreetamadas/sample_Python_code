import os
import random
import numpy as np


imgspath = "/mnt/data/"


def Train_Val_Test():

    images = []
    for i in os.listdir(imgspath):
        temp = path+i
        images.append(temp)
	
    random.seed(1000)
    random.shuffle(images)
    train,validate,test = np.split(images,[int(.6*len(images)), int(.8*len(images))])

    with open('/mnt/data/obj/train1.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    with open('/mnt/data/obj/validate1.txt', 'w') as f:
        for item in validate:
            f.write("%s\n" % item)        
    with open('/mnt/data/obj/test1.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)
