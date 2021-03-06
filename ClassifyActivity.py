
# coding: utf-8

# <img style="float: right;" alt="" width=250 src="http://pydata.org/berlin2015/static/img/PyDataBerlin-2015.png" />
# 
# # Running, walking, sitting or biking?
# ## Motion prediction with acceleration and rotationrates
# 
# A lot of devices can measure acceleration and rotation rates. With the right features, Machine Learning can predict, weather you are sitting, running, walking or going by bike.
# 
# * Talk by [Paul Balzer](http://trustme.engineer), MechLab Engineering
# * PyData Berlin 2015, 29. - 30.05.2015, Betahaus Berlin
# 

# In[210]:

from IPython.display import VimeoVideo
VimeoVideo('125699039', width=16*40, height=9*40)


# ## Load the stuff we need

# In[211]:

import pandas as pd
print('Pandas Version:\t%s' % pd.__version__)
import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')


# ## Load some sensor data with labeled activities
# 
# the `activitydata.csv` is prepared by the `prepData.ipynb`

# In[212]:

data = pd.read_csv('activitydata.csv', index_col=['timestamp'], parse_dates=True)


# In[213]:

dt = 1.0/50.0 # the activities were with 50Hz
data.index = np.arange(0, len(data)*dt, dt)
data.index.name='time'


# In[214]:

data.head(5)


# In[215]:

activities = data.groupby('activity')


# In[216]:

print np.round(activities.describe()[['accelerationX','motionRotationRateX']], decimals=2)


# To do some machine learning on that, we need features, which describe the activity best.

# # Features
# 
# ## Absolute Acceleration
# 
# absolute acceleration: $|a|=\sqrt{a_x^2 + a_y^2 + a_z^2}$
# 
# to get rid of the orientation of the device

# In[217]:

def absacc(row):
    return np.sqrt(row['accelerationX']**2 + row['accelerationY']**2 + row['accelerationZ']**2)


# In[218]:

data['absacc'] = data.apply(absacc, axis=1)

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.absacc[data.activity=='running'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('running'); axes[0,0].set_ylabel('|a| in $g$');
data.absacc[data.activity=='walking'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('walking');
data.absacc[data.activity=='cycling'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('biking'); axes[1,0].set_ylabel('|a| in $g$');
data.absacc[data.activity=='standing'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('standing');
plt.tight_layout()
plt.savefig('abs_acc_activities.pdf')


# ## Max-Min Difference of absolute Acceleration
# 
# difference between maximum and minimum acceleration with `rolling_max` and `rolling_min` 

# In[219]:

ws=1.0/dt # Window Size


# In[220]:

data['accmax'] = pd.rolling_max(data['absacc'], ws)
data['accmin'] = pd.rolling_min(data['absacc'], ws)

data['accmaxmindiff'] = data.accmax - data.accmin


# ### What's that `rolling_min` and `rolling_max`?

# In[221]:

fig, ax = plt.subplots()
data.absacc[data.activity=='running'].plot(ax=ax, label='acc')
data.accmax[data.activity=='running'].plot(ax=ax, label='rolling_max')
data.accmin[data.activity=='running'].plot(ax=ax, label='rolling_min')
plt.legend()
plt.ylabel('acc in $g$')

plt.tight_layout()
plt.savefig('abs_acc_window.pdf')


# In[222]:

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.accmaxmindiff[data.activity=='running'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('running'); axes[0,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='walking'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('walking');
data.accmaxmindiff[data.activity=='cycling'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('biking'); axes[1,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='standing'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('standingsitting');


# ### Let's take a look at these features

# In[223]:

activities = data.groupby('activity')


# In[224]:

fig, ax = plt.subplots()
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.absacc, marker='o', linestyle='', alpha=0.8, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'$|a|$ in [$g$]')


# `Running` and `standingsitting` can be seperated pretty good, but `walking` and `biking` is pretty much overlapped. We need some more features to seperate them.

# ## Amplitude of Rotation Rate
# 
# Let's take a look at the rotation rate of the device

# In[225]:

fig, ax = plt.subplots()
data.motionRotationRateX[data.activity=='walking'].plot(ax=ax, label='walking')
data.motionRotationRateX[data.activity=='cycling'].plot(figsize=(12,4), ax=ax, label='bike')
#plt.xlim(50,80)
plt.legend(loc='best')


# ## Fourier Transform of Rotation Rates
# 
# Periodic signals? Hm. Sounds like FFT. [Tutorial here](http://nbviewer.ipython.org/github/balzer82/FFT-Python/blob/master/FFT-Tutorial.ipynb)
# 
# ![FFT](http://www.cbcity.de/wp-content/uploads/2013/08/Time-2-Frequency-Domain-FFT.gif)

# In[226]:

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

# In[227]:

plt.figure(figsize=(6, 3.6))
colors = sns.color_palette()#['#FF6700', '#CAF278', '#3E3D2D', '#94C600']
for i, act in enumerate(['running','cycling','walking']):
    f, Y = fft_amplitude(data['motionRotationRateX'][data.activity==act], kind='full')
    plt.bar(f, Y, width=0.13, label=act, color=colors[i], alpha=0.8, align='center')

plt.xlabel(r'Frequency [$Hz$]')
plt.xlim(0, 10)
plt.ylabel(r'Amplitude [$rad/s$]')
plt.legend()

plt.tight_layout()
plt.savefig('fft_amplitude_rotationrate.pdf')


# OK, that is a good feature, so let's use it for the whole dataset with `rolling_apply`.

# In[228]:

data['fftamppeak'] = pd.rolling_apply(data['motionRotationRateX'], ws, fft_amplitude)


# In[229]:

fig, ax = plt.subplots()
data.fftamppeak[data.activity=='walking'].plot(ax=ax, label='walking')
data.fftamppeak[data.activity=='cycling'].plot(ax=ax, label='bike')
#plt.xlim(12,20)
plt.legend(loc='best')
plt.ylabel(r'fft amplitude peak in [$rad/s$]')


# # Let's do some Machine Learning on that

# To learn the basics, please take a look at this awesome tutorial from Jake: https://vimeo.com/80093925
# 
# Because of the `rolling_` functions, there is overlap between the activity features and the labels, corresponding to it. We have to delete some rows (length of window), before using a classifier.

# In[230]:

#data.drop(data[data.activity=='running'].iloc[0:int(ws)-1].index, inplace=True)
#data.drop(data[data.activity=='walking'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='cycling'].iloc[0:int(ws/2)-1].index, inplace=True)
data.drop(data[data.activity=='standing'].iloc[0:int(ws)-1].index, inplace=True)


# In[231]:

data.dropna(inplace=True)


# Now we can take a look at the features
# 
# ## Features

# In[232]:

activities = data.groupby('activity')

fig, ax = plt.subplots(figsize=(10,6))
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.fftamppeak, marker='o', linestyle='', alpha=0.8, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'fft amplitude peak in [$rad/s$]')
plt.xlim(-0.5,4)
plt.ylim(-0.5,4)


# # Scikit-Learn is our friend

# In[233]:

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing


# In[234]:

labels = data['activity'].values
np.shape(labels)


# In[235]:

featurevector = ['accmaxmindiff','fftamppeak']

features = data[featurevector].values
np.shape(features)


# ### Split in training and test

# In[236]:

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

np.shape(labels_test)


# ## Support Vector Classifier

# In[237]:

SVClassifier = SVC(kernel='linear').fit(features_train, labels_train)


# In[238]:

labels_predict_SVC = SVClassifier.predict(features_test)


# ### Accuracy

# In[239]:

print('%.3f%% accuracy' % (100.0*accuracy_score(labels_predict_SVC, labels_test)))


# In[240]:

cm = confusion_matrix(labels_predict_SVC, labels_test, labels=data.activity.unique())
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# In[241]:

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm_normalized, annot=True, fmt=".4f",
            cmap='Blues', square=True,
            xticklabels=data.activity.unique(),
            yticklabels=data.activity.unique())
ax.set_xlabel('Predicted Activity')
ax.set_ylabel('True Activity', )
plt.tight_layout()
plt.savefig('confusionmatrix.pdf')


# ### Plot decision boundaries
# 
# First we need to convert the labels (strings) to numerical values. There the `LabelEncoder` comes in...

# In[242]:

le = preprocessing.LabelEncoder()
le.fit(labels_predict_SVC)
list(le.classes_)


# In[243]:

colors=sns.color_palette()


# In[244]:

fig, ax = plt.subplots(figsize=(5,4))
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.fftamppeak, marker='o', linestyle='', alpha=0.9, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'FFT amplitude peak in [$rad/s$]')
plt.xlim(-0.5,3.5)
plt.ylim(-0.5,4)

xx, yy = np.meshgrid(np.arange(-0.5, 4, 0.01), np.arange(-0.5, 4, 0.01))
Z = SVClassifier.predict(np.c_[xx.ravel(), yy.ravel()])
Zn = le.transform(Z).reshape(xx.shape)
plt.contourf(xx, yy, Zn, colors=('b','k','g','k','r','k','k','purple'), alpha=0.2)
#plt.title('Decision Boundaries of Support Vector Classifier with features for 4 different activities')

plt.tight_layout()
plt.savefig('SVClassifier.pdf')


# In[245]:

SVClassifier.predict([1.0, 1.5])


# # Save the Classifier

# In[246]:

from sklearn.externals import joblib


# In[247]:

joblib.dump(SVClassifier, 'SVClassifier.pkl') 


# # Thank you for attention
# 
# Questions? [@Balzer82](https://twitter.com/balzer82)

# In[ ]:



