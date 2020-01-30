
#############################################################################################################
##### creating checksum of a file ###
# google : checksum of a pickle file ; pickle file md5sum
# https://stackoverflow.com/questions/5417949/computing-an-md5-hash-of-a-data-structure/5418117

## load original model
filename =  '/home/User/path/final_model.sav'   # .pkl or .sav file
loaded_model = pickle.load(open(filename, 'rb'))

## add version info & dump
ver = "version 1.0"
pickle.dump((loaded_model, ver), open("/home/Users/path2/final_model.sav", 'wb'))

## check if version info can be read : LOAD THE FILE
reload_model, version = pickle.load(open("/home/Users/path2/final_model.sav", 'rb'))

## GET CHECKSUM
import hashlib, random 
print(hashlib.md5(reload_model).hexdigest())

