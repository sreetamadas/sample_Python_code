## plotting methods
# http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html
# https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch04.html

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(filename)
# check datatype of the datetime column
print(df.dtypes)
# set it as datetime
df = pd.to_datetime(df['dateTime'])    # df['dateTime'] == df.dateTime

# set the datetime col as index for plotting
df = df.set_index('dateTime')


# see list of available styles for plots (font, line & point types)
print(plt.style.available)
# set style
plt.style.use('fivethirtyeight')   # fivethirtyeight  ; ggplot



### LinePlot
# METHOD 1:
# describe graph labels
ax = df.plot(color="blue", figsize=(8,3), linewidth=2, fontsize=6, marker='o')  # or, # plt.figure(figsize=(20,2), dpi=100)
# ax = df.plot(colormap='Dark2', figsize=(14, 7))  # use the colormap option to avoid color repetition for multiple lines in the plot
ax.set_xlabel('Date')
ax.set_ylabel('dependent variable Y')
ax.set_title('Y vs time')
# add horizontal & vertical lines
ax.axvline(x='2017-12-25', color='red', linestyle='--')
ax.axhline(y=100, color='green', linestyle='--')
# highlight a region
ax.axvspan('2017-12-01', '2018-01-01', color='red', alpha=0.3)
ax.axhspan(8, 6, color='green', alpha=0.3)
# show the plot
plt.show()

# METHOD 2:
mc['Date'] = pandas.to_datetime(mc.Date)
plt.figure(figsize=(14,2), dpi=100) # (width, height)
plt.plot(mc['Date'], mc['FracPower'], linewidth=0.1, color='black')
plt.axhline(y=100, linewidth=1, color='r')
plt.ylabel('Frac Power')
plt.show()




### area chart (to plot multiple time series)
# does area chart display cumulative values by default ?
ax = df.plot.area(figsize=(12, 4), fontsize=14)
plt.show()



### boxplot
ax1 = df.boxplot(fontsize=6, vert=False)  # vert specifies whether to plot vertically or horizontally
ax1.set_xlabel('v1')
ax1.set_ylabel('values')
ax1.set_title('Boxplot values of your data var. v1')
plt.show()

## box-plot of multiple groups (data by day-of-week)
from datetime import datetime
df['Day'] = pandas.to_datetime(df['Day']) 
df['day_of_week2'] = df['Day'].dt.weekday_name
df['day_of_week'] = df['Day'].dt.dayofweek
tmp = pandas.DataFrame(df[['day_of_week','y']])
fig, ax = plt.subplots(figsize=(10,8))
plt.suptitle('')
tmp.boxplot(column=['total_kWh'], by='day_of_week', ax=ax)
plt.show()



# barplot: distribution of data in different bins (create a df with labels/bins & corrs. values)
counts = pandas.DataFrame(pandas.value_counts(df['fac']) * 100/df.shape[0])
# df.shape[0] counts num_rows in df; df.shape[1] counts num_cols in df
counts = counts.reindex(["0","L","M","H","vH"])
ax = counts.plot(kind='bar', width=0.5, color='black', align='center', alpha=0.5)
ax.set_ylabel('% of total time')
ax.set_xlabel('power level')
ax.legend_.remove()
plt.show()



### histogram
ax2 = df.plot(kind='hist', bins=100)
ax2.set_xlabel('v1')
ax2.set_ylabel('Frequency of values in data')
ax2.set_title('Histogram of data: 100 bins')
plt.show()


### kernel density plots
ax3 = df.plot(kind='density', linewidth=2)
ax3.set_xlabel('v1')
ax3.set_ylabel('Density values')
ax.legend(fontsize=18)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=6)  ## specify fontsize & location of legend
ax3.set_title('Density plot')
plt.show()


### show lineplots for multiple series & their summaries
ax = df.plot(colormap='Dark2', figsize=(14, 7))
# https://matplotlib.org/examples/color/colormaps_reference.html
df_summary = df.describe()
# Specify values of cells in the table
ax.table(cellText=df_summary.values, 
          # Specify width of the table
          colWidths=[0.3]*len(df.columns), 
          # Specify row labels
          rowLabels=df_summary.index, 
          # Specify column labels
          colLabels=df_summary.columns, 
          # Specify location of the table
          loc='top') 
plt.show()


### FACETING multiple line plots, especially for different cols with different scales  ###
df.plot(subplots=True,
                linewidth=0.5,
                layout=(2, 4),   # specifies no. of rows & cols in the figure
                figsize=(16, 10),
                sharex=False,
                sharey=False)
plt.show()



### heatmap of correlation matric
corr_mat = df.corr(method='pearson')
import seaborn as sns
#sns.heatmap(corr_mat, annot=True, linewidths=0.4, annot_kws={"size": 10})
sns.heatmap(corr_mat, mask=np.zeros_like(corr_mat, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
square=True, ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# clustermap to group together similar columns (using hierarchical clustering)
sns.clustermap(corr_mat, row_cluster=True, col_cluster=True,)
plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()



### scatterplots, with points colored by group 
import numpy as np
df['col'] = np.where(df['Y'] == 1.0 , 'red', 'yellow')  # red if Y=1; yellow otherwise

df1 = df[['dateTime','Y', 'col']]
df1['dateTime'] = pd.to_datetime(df1['dateTime'])
df1['t1'] = df1.dateTime.astype(np.int64)

fig, ax = plt.subplots(figsize=(12,3))
ax.scatter(df1['t1'],df1['Y'],c=df1['col'], marker = 'o') #, cmap = cm.jet )
#ax2.set_xlabel('date')
#ax2.set_ylabel('temperature')
plt.show()



### scattermatrix (similar to pairwise plots of all pairs of colunms in a df; useful only for numeric data )###
pd.scatter_matrix(df, c=y, figsize=[10,10], s=20, marker='o']    # df has only the input features, y is the target variable
# c is for color, s is for marker size
                  
                  
## seaborn countplot - useful for categorical data
plt.figure()
sns.countplot(x='x1', hue='y', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
# sample data                  
# sr  x1   x2   y
# 1   no   ..   A 
# 2   yes  ..   A
...
# 20  no   ..   B
...                  

                  
                  
                  
#### save a plot to an external file
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
ada.fit(X, Y)
importances_ada = ada.feature_importances_  
## REALTIVE FEATURE IMPORTANCE ##
FI_ada = 100.0 * (importances_ada / importances_ada.max())
labels = mytrain.columns
indices = np.argsort(FI_ada)[::-1]
# Plot the feature importances of the forest
plt.figure()
plt.title("Relative Feature importances")
plt.bar(labels[indices], FI_ada[indices], color="black", align="center")   # yerr=std[indices], range(mytrain.shape[1])
plt.xticks(labels[indices], rotation='vertical')
#plt.xlim([-1, mytrain.shape[1]])
#plt.show()
plt.savefig('varImp_adaB_iter_'+str(i)+'.png', bbox_inches='tight')
plt.close()
                  
                  
##################################################                  
def func_plot():
    ...
    plt.plot(temp) #, marker='o', markersize=2) #, label=str(m)) 
    plt.scatter(x=x,y=y,c='r')
    plt.legend(loc='best')
                  
plt.subplot(221)
func_plot(pd.np.array(data_file1.iloc[:,useful_col[0]]).reshape(len(data_file1)),pat_id

