#####  dataframe manipulations  #####


####  concatenation  ####
# ADD 1 df below the other
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=0)  # axis=0 means concatenating by row

## ADD dataframes by column
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=1)  # axis=1 means concatenating by column



####  merge  : inner join, outer join, etc




#### array (row) to df, 
df=pd.DataFrame(array).T     # T implies taking transpose


####   append, extend




loc & iloc;



####   passing df row to a function through loop
for i in range(0,df.shape[0]):  
    total=0
    X_score = user_defined_python_function(df.loc[i, 'columnX'])

