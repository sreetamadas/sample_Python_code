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


### boxplot
ax1 = df.boxplot()
ax1.set_xlabel('v1')
ax1.set_ylabel('values')
ax1.set_title('Boxplot values of your data var. v1')
plt.show()


### histogram
ax2 = df.plot(kind='hist', bins=100)
ax2.set_xlabel('v1')
ax2.set_ylabel('Frequency of values in your data')
ax2.set_title('Histogram of data: 100 bins')
plt.show()


### kernel density plots

