
# coding: utf-8

# In[217]:

import pandas as pd
print('Pandas Version:\t%s' % pd.__version__)
import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[218]:

data = pd.read_csv('activitydata.csv', index_col=['timestamp'], parse_dates=True)


# In[219]:

dt = 0.02 # s
data.index = np.arange(0, len(data)*dt, dt)
data.index.name='time'


# In[220]:

data.head(5)


# In[221]:

activities = data.groupby('activity')


# In[ ]:




# In[ ]:




# In[ ]:




# # Features
# 
# ## Absolute Acceleration
# 
# absolute acceleration: $|a|=\sqrt{a_x^2 + a_y^2 + a_z^2}$
# 
# to get rid of the orientation of the device

# In[222]:

def absacc(row):
    return np.sqrt(row['accelerationX']**2 + row['accelerationY']**2 + row['accelerationZ']**2)


# In[223]:

data['absacc'] = data.apply(absacc, axis=1)

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.absacc[data.activity=='Joggen'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('Joggen'); axes[0,0].set_ylabel('acc in $g$');
data.absacc[data.activity=='Laufen'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('Laufen');
data.absacc[data.activity=='Radfahren'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('Radfahren'); axes[1,0].set_ylabel('acc in $g$');
data.absacc[data.activity=='Sitzen'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('Sitzen');


# ## Max-Min Difference of absolute Acceleration
# 
# difference between maximum and minimum acceleration with `rolling_max` and `rolling_min` 

# In[224]:

ws=0.5/dt # Window Size


# In[ ]:




# In[225]:

data['accmax'] = pd.rolling_max(data['absacc'], ws)
data['accmin'] = pd.rolling_min(data['absacc'], ws)

data['accmaxmindiff'] = data.accmax - data.accmin


# In[226]:

fig, ax = plt.subplots()
data.absacc[data.activity=='Joggen'].plot(ax=ax, label='acc')
data.accmax[data.activity=='Joggen'].plot(ax=ax, label='rolling_max')
data.accmin[data.activity=='Joggen'].plot(ax=ax, label='rolling_min')
plt.legend()


# In[227]:

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.accmaxmindiff[data.activity=='Joggen'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('Joggen'); axes[0,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='Laufen'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('Laufen');
data.accmaxmindiff[data.activity=='Radfahren'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('Radfahren'); axes[1,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='Sitzen'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('Sitzen');


# ### Let's take a look at these features

# In[228]:

activities = data.groupby('activity')


# In[229]:

fig, ax = plt.subplots()
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.absacc, marker='o', linestyle='', alpha=0.8, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'$|a|$ in [$g$]')


# Joggen and Sitzen can be seperated pretty good, but Laufen and Radfahren is pretty much overlapped. We need some more features to seperate them.

# ## Amplitude of Rotation Rate Spectrum
# 
# Let's take a look at the rotation rate of the device

# In[230]:

fig, ax = plt.subplots()
data.motionRotationRateX[data.activity=='Laufen'].plot(ax=ax, label='walking')
data.motionRotationRateX[data.activity=='Radfahren'].plot(figsize=(12,4), ax=ax, label='bike')
#plt.xlim(50,80)
plt.legend(loc='best')


# Periodic signals? Hm. Sounds like FFT. [Tutorial here](http://nbviewer.ipython.org/github/balzer82/FFT-Python/blob/master/FFT-Tutorial.ipynb)
# 
# ![FFT](http://www.quickmeme.com/img/b1/b151ee77782b3e72d4c548f52563e1a81f6c4828ba56d37c392f09b469e9d5e3.jpg)

# In[231]:

def fft_amplitude(s, kind='peak'):
    
    # don't forget the windowing to get rid of the leakage effect
    hann = np.hanning(len(s)) 
    
    # do the FFT with Hanning Window
    Yhann = np.fft.fft(hann*s)
    
    N = len(Yhann)/2+1
    Y = 2.0*np.abs(Yhann[:N])/N # right half is enough info
    
    # frequency axis, if needed
    fa = 1.0/dt
    f = np.linspace(0, fa/2.0, N, endpoint=True)
    
    if kind=='peak':
        return np.max(Y) # just return the maximum peak amplitude
    elif kind=='periodicity':
        return np.max(Y) / np.mean(Y) # return periodicity
    elif kind=='full':
        return f, Y # return the full spectrum    


# ### Let's take a look

# In[232]:

plt.figure(figsize=(9, 4.5))
colors = ['#FF6700', '#CAF278', '#3E3D2D', '#94C600']
for i, act in enumerate(['Laufen','Radfahren']):
    f, Y = fft_amplitude(data['motionRotationRateX'][data.activity==act], kind='full')
    plt.bar(f, Y, width=0.1, label=act, color=colors[i], alpha=0.8, edgecolor='k', align='center')

plt.xlabel(r'Frequenz [$Hz$]')
plt.xlim(0, np.max(f))
plt.ylabel(r'Amplitude [$rad/s$]')
plt.legend()


# OK, that is a good feature, so let's use it for the whole dataset.

# In[233]:

data['fftamppeak'] = pd.rolling_apply(data['motionRotationRateX'], ws, fft_amplitude)


# In[234]:

fig, ax = plt.subplots()
data.fftamppeak[data.activity=='Laufen'].plot(ax=ax, label='walking')
data.fftamppeak[data.activity=='Radfahren'].plot(ax=ax, label='bike')
plt.xlim(12,20)
plt.legend(loc='best')
plt.ylabel('fft amplitude max')


# Because of the `rolling_` functions, there is overlap between the activity features and the labels, corresponding to it. We have to delete some rows (length of window), before using a classifier.

# In[235]:

data.drop(data[data.activity=='Joggen'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='Laufen'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='Radfahren'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='Sitzen'].iloc[0:int(ws)-1].index, inplace=True)


# Now we can take a look at the features

# In[236]:

activities = data.groupby('activity')

fig, ax = plt.subplots(figsize=(10,6))
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.fftamppeak, marker='o', linestyle='', alpha=0.8, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'')
plt.xlim(-1,4)
plt.ylim(-0.5,5)


# In[237]:

# Drop every row, which has no value
#data.dropna(inplace=True)


# # Classifier

# In[238]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[239]:

labels = data['activity'].values
np.shape(labels)


# In[240]:

featurevector = ['accmaxmindiff','fftamppeak']
#featurevector = ['accelerationX', 'accelerationY', 'accelerationZ']

features = data[featurevector].values
np.shape(features)


# In[241]:

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.8, random_state=12)

np.shape(features_train)


# ## Decision Tree

# In[265]:

DTclassifier = tree.DecisionTreeClassifier(max_depth=3).fit(features_train, labels_train)


# In[266]:

labels_predict_DT = DTclassifier.predict(features_test)


# In[267]:

DTclassifier.predict([0.0399, 0.0134])


# In[268]:

accuracy_score(labels_predict_DT, labels_test)


# In[269]:

importances = DTclassifier.feature_importances_
featureimportance = pd.DataFrame(index=featurevector, data=importances, columns=['Importance']).sort('Importance', ascending=False).plot(kind='bar', rot=20)


# ### Export a graph of the Decision Tree

# Export for visualization. Later we can use Graphviz’s dot tool to create a PDF file (or any other supported file type): `dot -Tpdf DTClassifier.dot -o DTClassifier.pdf`

# In[270]:

from sklearn.externals.six import StringIO
with open("DTClassifier.dot", 'w') as f:
    f = tree.export_graphviz(DTclassifier, out_file=f)
    
get_ipython().system(u'dot -Tpdf DTClassifier.dot -o DTClassifier.pdf')


# ## Random Forest

# In[271]:

RFclassifier = RandomForestClassifier().fit(features_train, labels_train)


# In[272]:

labels_predict_RF = RFclassifier.predict(features_test)


# In[273]:

accuracy_score(labels_predict_RF, labels_test)


# In[274]:

importances = RFclassifier.feature_importances_
featureimportance = pd.DataFrame(index=featurevector, data=importances, columns=['Importance']).sort('Importance', ascending=False).plot(kind='bar', rot=20)


# In[ ]:




# # Save the Classifier

# In[275]:

import pickle


# In[276]:

with open('./RFclassifier.pkl', 'wb') as fid:
    pickle.dump(RFclassifier, fid)
with open('./DTclassifier.pkl', 'wb') as fid:
    pickle.dump(DTclassifier, fid)


# In[ ]:



