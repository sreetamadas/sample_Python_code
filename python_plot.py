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

# describe graph labels
ax = df.plot(color="blue", figsize=(8,3), linewidth=2, fontsize=6)
ax.set_xlabel('Date')
ax.set_ylabel('dependent variable Y')
ax.set_title('Y vs time')
plt.show()

