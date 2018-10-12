## split a string by some character : the resultants are also strings
1st_element_of_string = str(somestring.split('_')[0])


## getting numeric data from a text file
f=open("/home/user/Desktop/data/file.txt")
data=f.read()
data_new=data.splitlines()
for k in range(len(data_new)):
    string_data=data_new[k].split(' ')                ## split convert numeric data to string
    string_to_float=[float(i) for i in string_data]   ## convert back to numeric
    dat.append(string_to_float)
data_file=pd.DataFrame(dat)



