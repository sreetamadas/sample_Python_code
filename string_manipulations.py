## split a string by some character : the resultants are also strings
first_element_of_string = str(some_string.split('_')[0])


## select a portion of a string by position no.
ID = filename[2:5]


## getting numeric data from a text file
f=open("/home/user/Desktop/data/file.txt")
data=f.read()
data_new=data.splitlines()
for k in range(len(data_new)):
    string_data=data_new[k].split(' ')                ## split convert numeric data to string
    string_to_float=[float(i) for i in string_data]   ## convert back to numeric
    dat.append(string_to_float)
data_file=pd.DataFrame(dat)



## combine many results as string to print as output 
f= 'status=' + str(value) + ',' + 'X = ' + str(round(frac_result[0],1)) + ',' + 'Y' + '=' + str(int(rate)) 
print(f)


