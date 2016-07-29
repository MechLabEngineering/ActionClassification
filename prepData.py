
# coding: utf-8

# In[1]:

import pandas as pd
import os


# # Prepare the logfiles to learn with the raw data
# 
# the `.log` files were recorded with an old version of the great [SensorLog App for iOS](https://itunes.apple.com/us/app/sensorlog/id388014573?mt=8). Thanks Bernd Thomas!

# In[2]:

csvs = [files for files in os.listdir('./') if files.endswith('.log')]

pd.read_csv(csvs[1]).head(5)


# ## Dump activities in one file with labels
# 
# just a snipped of the whole data

# In[3]:

von = 6.0
bis = 14.0


# In[4]:

values = ['accelerationX','accelerationY','accelerationZ','motionRotationRateX','motionRotationRateY', 'motionRotationRateZ']

# Write Header in CSV
with open('activitydata.csv', 'wb') as f:
    f.write("timestamp," + ",".join(values) + ',activity\n')

# Open all Activity csvs and get data
for csv in csvs:
    print('Lade %s' % csv)
    data = pd.read_csv(csv, index_col='recordtime')
    data = data[(data.index<=bis) & (data.index>=von)] # equal time ranges
    
    data.index = data.time
    data = data[values] # just keep relevant values

    data['activity'] = csv.split('_')[1] # label data
    
    with open('activitydata.csv', 'a') as f:
        data.to_csv(f, header=False, float_format='%.6f')


# In[ ]:



