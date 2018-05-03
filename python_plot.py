## plotting methods

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
plt.style.use('fivethirtyeight')


### LinePlot
# describe graph labels
ax = df.plot(color="blue", figsize=(8,3), linewidth=2, fontsize=6)
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


### area chart (to plot multiple time series)
# does area chart display cumulative values by default ?
ax = df.plot.area(figsize=(12, 4), fontsize=14)
plt.show()


### boxplot
ax1 = df.boxplot()
ax1.set_xlabel('v1')
ax1.set_ylabel('values')
ax1.set_title('Boxplot values of your data var. v1')
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


### faceting multiple line plots, especially for different cols with different scales
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
sns.heatmap(corr_mat, annot=True, linewidths=0.4, annot_kws={"size": 10})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# clustermap to group together similar columns (using hierarchical clustering)
sns.clustermap(corr_mat, row_cluster=True, col_cluster=True,)
plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()



