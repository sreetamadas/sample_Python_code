# copy files from yoloGC-10 directory to tensorflow dataset

import shutil

# location of train.txt, val.txt & test.txt
listpath = '/mnt/tiny_yolo/obj/'

# location for images & initial annotations (xml or txt)
#in_img = '/mnt/Data/GC10_Updated/'
in_anno = '/mnt/Data/label/'

# final data folder
out = '/mnt/tensorflow/workspace2/training_demo/images/'


# test
f = open(listpath+"test.txt") #"./data/dataset_ids/test.txt")
data = f.read()
test_dat = data.splitlines()

for i in test_dat:
    infile1 = i  #in_img + 'test/' + i + '.jpg'
    i = i.split(".jpg")[0]
    #infile2 = i + '.txt'  
    infile2 = in_anno + i.split("/")[-1] + '.xml' #in_anno + i + '.xml'
    out_file = out + 'test/'            #+ i + '.jpg'
    #shutil.copy2(infile1, out_file)
    shutil.copy2(infile2, out_file)


# train
f = open(listpath+"train.txt") #"./data/dataset_ids/train.txt")
data = f.read()
train_dat = data.splitlines()
#len(train_dat)

for i in train_dat:
    infile1 = i #in_img + 'train/' + i + '.jpp'
    i = i.split(".jpg")[0]
    #infile2 = i + '.txt' 
    infile2 = in_anno + i.split("/")[-1] + '.xml' #in_anno + i + '.xml'
    out_file = out + 'train/'
    #shutil.copy2(infile1, out_file)
    shutil.copy2(infile2, out_file)



# val
f = open(listpath+"validate.txt") #"./data/dataset_ids/val.txt")
data = f.read()
val_dat = data.splitlines()

for i in val_dat:
    infile1 = i #in_img + 'train/' + i + '.jpp'
    i = i.split(".jpg")[0]
    #infile2 = i + '.txt' 
    infile2 = in_anno + i.split("/")[-1] + '.xml' #in_anno + i + '.xml'
    out_file = out + 'val/'
    #shutil.copy2(infile1, out_file)
    shutil.copy2(infile2, out_file)


