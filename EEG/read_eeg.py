## read EEG data

import pandas as pd
import numpy as np
import os
import pyedflib

# the annotation file 'y' has last entry at 79500
# while the signal 'x' has upto 7950000
# why this discrepancy ?
# indirect solution: read in the data & the annotation using EDF browser, then export the annotation as txt/csv file;
# then read it in using python

######################################################################
# https://stackoverflow.com/questions/47053080/import-edf-file-directly-from-online-archive-in-python
# https://www.researchgate.net/post/Is_it_possible_to_convert_EEG_dataset_into_csv_file_and_then_analyse_using_R_tool_or_weka

# https://www.researchgate.net/post/Does_anyone_know_how_to_open_edf_files_in_python
# http://forrestbao.blogspot.com/2014/07/reading-physiology-data-in-edf-format.html
# https://martinos.org/mne/dev/manual/io.html


#filepath = raw_input("enter fullPath & filename: ")  # C:\\Users\\Desktop\\data\\filename.xls
file_name = "C:\\Users\\Desktop\\data\\EEG\\data\\SC4001E0-PSG.edf"
f = pyedflib.EdfReader(file_name)

#####################################################################
#https://stackoverflow.com/questions/38145941/transform-edf-file-on-python-in-txt
# https://github.com/holgern/pyedflib/blob/master/demo/readEDFFile.py

print("file duration: %i seconds" % f.file_duration)
print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
print("patientcode: %s" % f.getPatientCode())
print("gender: %s" % f.getGender())
print("birthdate: %s" % f.getBirthdate())
print("patient_name: %s" % f.getPatientName())
print("patient_additional: %s" % f.getPatientAdditional())
print("admincode: %s" % f.getAdmincode())
print("technician: %s" % f.getTechnician())
print("equipment: %s" % f.getEquipment())
print("recording_additional: %s" % f.getRecordingAdditional())
print("datarecord duration: %f seconds" % f.getFileDuration())
print("number of datarecords in the file: %i" % f.datarecords_in_file)
print("number of annotations in the file: %i" % f.annotations_in_file)

#channel = 3
#print("\nsignal parameters for the %d.channel:\n\n" % channel)

#print("label: %s" % f.getLabel(channel))
#print("samples in file: %i" % f.getNSamples()[channel])
#print("physical maximum: %f" % f.getPhysicalMaximum(channel))
#print("physical minimum: %f" % f.getPhysicalMinimum(channel))
#print("digital maximum: %i" % f.getDigitalMaximum(channel))
#print("digital minimum: %i" % f.getDigitalMinimum(channel))
#print("physical dimension: %s" % f.getPhysicalDimension(channel))
#print("prefilter: %s" % f.getPrefilter(channel))
#print("transducer: %s" % f.getTransducer(channel))
#print("samplefrequency: %f" % f.getSampleFrequency(channel))

##########################################################################
#https://stackoverflow.com/questions/48489009/how-can-i-read-edf-event-file-with-python

#f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
n
#signal_labels = f.getSignalLabels()
#sigbufs = np.zeros((n,f.getNSamples()[0]))
#for i in np.arange(n):
#    sigbufs[i,:]=f.readSignal(i)

#return sigbufs

##########################################################3
signal_labels = f.getSignalLabels()
signal_labels

##############################################################
# The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep recordings, containing
# EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature.

# https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
# There are also (Z) sites: A "Z" (zero) refers to an electrode placed on the midline saggital plane of the skull,
# (FpZ, Fz, Cz, Oz) and is present mostly for reference/measurement points. 

#################################################################
f.getNSamples()

###################################################################
# read in the 1st channel
# https://github.com/holgern/pyedflib/blob/master/pyedflib/edfreader.py

eeg_1 = f.readSignal(0)
print('signal 1 no of data pts: ' + str(eeg_1.shape[0]))

########################################################################
# read separate annotation file

f2 = pyedflib.EdfReader("C:\\Users\\Desktop\\data\\EEG\\data\\SC4002EC-Hypnogram.edf")
f2.readAnnotations()

################################################################
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html
# https://stackoverflow.com/questions/25909984/missing-data-insert-rows-in-pandas-and-fill-with-nan

# create a new df to dave the signal & annotations
# https://cmdlinetips.com/2018/01/how-to-create-pandas-dataframe-from-multiple-lists/

#####################################################################
# create a new array with differences of values in col 1 of annotation file, then multiply each value by 10
# add another entry = (86400 - last entry) * 10

anno = f2.readAnnotations()
df = {'start_time':anno[0], 'interval':anno[1], 'anno':anno[2]}  # 'X':anno[1], 
df = pd.DataFrame(df)
df['interval'] = df['interval'] * 10
df['start_time'] = df['start_time'] * 10
df.head(5)

########################################################
# add intervals
df['interval'].sum()

new_index = range(eeg_1.shape[0])   #[0:x.shape[0]]
new_index

##########################################################
d = df.set_index(df['start_time']).reindex(new_index)
d = d.fillna(method='ffill')
d.tail(5)

##########################################################
## add the signal value to the dataframe

dd = pd.concat([d,pd.DataFrame(eeg_1)],axis=1)
dd

##############################################################
dd.groupby('anno').agg({'start_time': 'count'})

# 27600+198000 + 24000 + 4800 + 7695600 + 52800 + 547200 = 8550000

