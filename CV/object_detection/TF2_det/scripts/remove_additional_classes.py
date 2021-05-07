# -*- coding: utf-8 -*-


# remove extra classes from the dataset
import pandas as pd

def clean_df(path, file):
    # read the original csv with all classes
    file = path + file
    df = pd.read_csv(file)
    
    # list of classes to be removed
    remove_list = ['10_yaozhe','9_zhehen','8_yahen','7_yiwu','6_siban','5_youban']
    cleaned = df[~df['class'].isin(remove_list)]
    return cleaned



path = "/mnt/tensorflow/workspace2/training_demo/images/"
new = clean_df(path, "test_labels.csv")
new.to_csv(path+"test_labels_new.csv", index=False)

new = clean_df(path, "train_labels.csv")
new.to_csv(path+"train_labels_new.csv", index=False)

new = clean_df(path, "val_labels.csv")
new.to_csv(path+"val_labels_new.csv", index=False)
