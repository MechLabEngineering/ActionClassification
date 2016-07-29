
# coding: utf-8

# ## Load the stuff we need

# In[1]:

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
# use the [PAMAP2 Dataset](http://www.pamap.org/demo.html), with `subject106`

# In[2]:

data = pd.read_csv('./PAMAP2_Dataset/Protocol/subject106.dat', sep=' ',
                  names=['timestamp','activityID','heartrate', \
                         'IMU_hand_temp','IMU_hand_ax1','IMU_hand_ay1','IMU_hand_az1', \
                         'IMU_hand_ax2','IMU_hand_ay2','IMU_hand_az2', \
                         'IMU_hand_rotx','IMU_hand_roty','IMU_hand_rotz', \
                         'IMU_hand_magx','IMU_hand_magy','IMU_hand_magz', \
                         'IMU_hand_oru','IMU_hand_orv','IMU_hand_orw', 'IMU_hand_orx', \
                         'IMU_chest_temp','IMU_chest_ax1','IMU_chest_ay1','IMU_chest_az1', \
                         '','IMU_chest_ay2','IMU_chest_az2', \
                         'IMU_chest_rotx','IMU_chest_roty','IMU_chest_rotz', \
                         'IMU_chest_magx','IMU_chest_magy','IMU_chest_magz', \
                         'IMU_chest_oru','IMU_chest_orv','IMU_chest_orw', 'IMU_chest_orx', \
                         'IMU_ankle_temp','IMU_ankle_ax1','IMU_ankle_ay1','IMU_ankle_az1', \
                         'IMU_ankle_ax2','IMU_ankle_ay2','IMU_ankle_az2', \
                         'IMU_ankle_rotx','IMU_ankle_roty','IMU_ankle_rotz', \
                         'IMU_ankle_magx','IMU_ankle_magy','IMU_ankle_magz', \
                         'IMU_ankle_oru','IMU_ankle_orv','IMU_ankle_orw', 'IMU_ankle_orx'])


# In[3]:

dt = 1.0/100.0 # the activities were with 50Hz
#data.index = np.arange(0, len(data)*dt, dt)
#data.index.name='time'
data.index = data.timestamp


# In[4]:

data.head(5)


# In[5]:

data.dropna(subset = ['IMU_hand_temp', 'IMU_chest_temp', 'IMU_ankle_temp'], inplace=True)


# In[6]:

activitymap = {1: 'lying',
               2: 'sitting',
               3: 'standing',
               4: 'walking',
               5: 'running',
               6: 'cycling',
               7: 'Nordic walking',
               9: 'watching TV',
               10: 'computer work',
               11: 'car driving',
               12: 'ascending stairs',
               13: 'descending stairs',
               16: 'vacuum cleaning',
               17: 'ironing',
               18: 'folding laundry',
               19: 'house cleaning',
               20: 'playing soccer',
               24: 'rope jumping',
               0: 'other'}


# In[ ]:




# In[7]:

data['activity'] = data['activityID'].apply(activitymap.get)


# Just keep 4 activities

# In[8]:

data.drop(data.index[(data.activity!='standing') & (data.activity!='cycling') & (data.activity!='running') & (data.activity!='walking')], inplace=True)


# In[9]:

activities = data.groupby('activity')


# Cut off first and last 1000 items, because activity starts end ends

# In[10]:

data.drop(data[data.activity=='running'].iloc[:1000].index, inplace=True)
data.drop(data[data.activity=='running'].iloc[-1000:].index, inplace=True)
data.drop(data[data.activity=='walking'].iloc[:1000].index, inplace=True)
data.drop(data[data.activity=='walking'].iloc[-1000:].index, inplace=True)
data.drop(data[data.activity=='cycling'].iloc[:1000].index, inplace=True)
data.drop(data[data.activity=='cycling'].iloc[-1000:].index, inplace=True)
data.drop(data[data.activity=='standing'].iloc[:1000].index, inplace=True)
data.drop(data[data.activity=='standing'].iloc[-1000:].index, inplace=True)


# In[11]:

data.head()


# In[12]:

#print np.round(activities.describe()[['IMU_chest_ax1','IMU_chest_rotx']], decimals=2)


# To do some machine learning on that, we need features, which describe the activity best.

# # Features
# 
# ## Absolute Acceleration
# 
# absolute acceleration: $|a|=\sqrt{a_x^2 + a_y^2 + a_z^2}$
# 
# to get rid of the orientation of the device

# In[13]:

def absacc(row):
    return np.sqrt(row['IMU_chest_ax1']**2 + row['IMU_chest_ay1']**2 + row['IMU_chest_ay1']**2)/9.806


# In[14]:

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

# In[15]:

ws=1.0/dt # Window Size


# In[16]:

data['accmax'] = pd.rolling_max(data['absacc'], ws)
data['accmin'] = pd.rolling_min(data['absacc'], ws)

data['accmaxmindiff'] = data.accmax - data.accmin


# ### What's that `rolling_min` and `rolling_max`?

# In[17]:

fig, ax = plt.subplots()
data.absacc[data.activity=='running'].plot(ax=ax, label='acc')
data.accmax[data.activity=='running'].plot(ax=ax, label='rolling_max')
data.accmin[data.activity=='running'].plot(ax=ax, label='rolling_min')
plt.legend()
plt.ylabel('acc in $g$')

plt.tight_layout()
plt.savefig('abs_acc_window.pdf')


# In[18]:

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.accmaxmindiff[data.activity=='running'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('running'); axes[0,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='walking'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('walking');
data.accmaxmindiff[data.activity=='cycling'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('biking'); axes[1,0].set_ylabel('acc in $g$');
data.accmaxmindiff[data.activity=='standing'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('standingsitting');


# ### Let's take a look at these features

# In[19]:

activities = data.groupby('activity')


# In[20]:

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

# In[21]:

# Let's take a look
fig, axes = plt.subplots(nrows=2, ncols=2)
data.IMU_chest_rotx[data.activity=='running'].plot(ax=axes[0,0], ylim=(0,5)); axes[0,0].set_title('running'); axes[0,0].set_ylabel('rotationrate in $Deg/s$');
data.IMU_chest_rotx[data.activity=='walking'].plot(ax=axes[0,1], ylim=(0,5)); axes[0,1].set_title('walking');
data.IMU_chest_rotx[data.activity=='cycling'].plot(ax=axes[1,0], ylim=(0,5)); axes[1,0].set_title('cycling'); axes[1,0].set_ylabel('rotationrate in $Deg/s$');
data.IMU_chest_rotx[data.activity=='standing'].plot(ax=axes[1,1], ylim=(0,5)); axes[1,1].set_title('standing');


# ## Fourier Transform of Rotation Rates
# 
# Periodic signals? Hm. Sounds like FFT. [Tutorial here](http://nbviewer.ipython.org/github/balzer82/FFT-Python/blob/master/FFT-Tutorial.ipynb)
# 
# ![FFT](http://www.cbcity.de/wp-content/uploads/2013/08/Time-2-Frequency-Domain-FFT.gif)

# In[22]:

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

# In[23]:

plt.figure(figsize=(6, 3.6))
colors = sns.color_palette()#['#FF6700', '#CAF278', '#3E3D2D', '#94C600']
for i, act in enumerate(['running','cycling','walking']):
    f, Y = fft_amplitude(data['IMU_chest_rotz'][data.activity==act], kind='full')
    plt.bar(f[::20], Y[::20], width=0.15, label=act, color=colors[i], alpha=0.8, align='center')

plt.xlabel(r'Frequency [$Hz$]')
plt.xlim(0, 10)
plt.ylabel(r'Amplitude [$rad/s$]')
plt.legend()

plt.tight_layout()
plt.savefig('fft_amplitude_rotationrate.pdf')


# OK, that is a good feature, so let's use it for the whole dataset with `rolling_apply`.

# In[24]:

data['fftamppeak'] = pd.rolling_apply(data['IMU_chest_rotz'], 1.0*ws, fft_amplitude)


# In[25]:

fig, ax = plt.subplots()
data.fftamppeak[data.activity=='walking'].plot(ax=ax, label='walking')
data.fftamppeak[data.activity=='cycling'].plot(ax=ax, label='cycling')
#plt.xlim(12,20)
plt.legend(loc='best')
plt.ylabel(r'fft amplitude peak in [$rad/s$]')


# # Let's do some Machine Learning on that

# To learn the basics, please take a look at this awesome tutorial from Jake: https://vimeo.com/80093925
# 
# Because of the `rolling_` functions, there is overlap between the activity features and the labels, corresponding to it. We have to delete some rows (length of window), before using a classifier.

# In[26]:

data.drop(data[data.activity=='running'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='walking'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='cycling'].iloc[0:int(ws)-1].index, inplace=True)
data.drop(data[data.activity=='standing'].iloc[0:int(ws)-1].index, inplace=True)


# In[27]:

#data.dropna(inplace=True)


# Now we can take a look at the features
# 
# ## Features

# In[ ]:




# In[28]:

activities = data.groupby('activity')

fig, ax = plt.subplots(figsize=(10,6))
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.fftamppeak, marker='o', linestyle='', alpha=0.8, label=activity)

ax.legend(loc=2)
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'fft amplitude peak in [$rad/s$]')
#plt.xlim(-0.5,4)
#plt.ylim(-0.5,4)


# # Scikit-Learn is our friend

# In[29]:

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing


# In[30]:

labels = data['activity'].values
np.shape(labels)


# In[31]:

featurevector = ['accmaxmindiff','fftamppeak']

features = data[featurevector].values
np.shape(features)


# ### Split in training and test

# In[32]:

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

np.shape(labels_test)


# ## Support Vector Classifier
# 
# Load the trained on

# In[220]:

SVClassifier = SVC(kernel='rbf', C=1.0, gamma=0.5).fit(features_train, labels_train)


# In[221]:

labels_predict_SVC = SVClassifier.predict(features_test)


# ### Accuracy

# In[222]:

print('%.3f%% accuracy' % (100.0*accuracy_score(labels_predict_SVC, labels_test)))


# In[223]:

cm = confusion_matrix(labels_predict_SVC, labels_test, labels=data.activity.unique())
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# In[232]:

# Latex Table Line for Paper:
print('proposed & %.2f & %.2f & %.2f & %.2f \\\\' % (100.0*cm_normalized[3,3], 100.0*cm_normalized[0,0], 100.0*cm_normalized[1,1], 100.0*cm_normalized[2,2]))


# In[225]:

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

# In[226]:

le = preprocessing.LabelEncoder()
le.fit(labels_predict_SVC)
list(le.classes_)


# In[227]:

colors=sns.color_palette()


# In[235]:

fig, ax = plt.subplots(figsize=(6,4))
for activity, group in activities:
    ax.plot(group.accmaxmindiff, group.fftamppeak, marker='o', alpha=.6, linestyle='', label=activity)

legend = ax.legend(loc=1)
    
plt.xlabel(r'$|a|_{max} - |a|_{min}$ in [$g$]')
plt.ylabel(r'FFT amplitude peak in [$rad/s$]')
plt.xlim(-0.1,6.9)
plt.ylim(-0.1,2.4)

xx, yy = np.meshgrid(np.arange(-0.5, 7, 0.01), np.arange(-0.5, 3, 0.01))
Z = SVClassifier.predict(np.c_[xx.ravel(), yy.ravel()])
Zn = le.transform(Z).reshape(xx.shape)
plt.contourf(xx, yy, Zn, colors=('b','k','g','k','r','k','k','purple'), alpha=0.3)
#plt.title('Decision Boundaries of Support Vector Classifier with features for 4 different activities')

plt.tight_layout()
plt.savefig('SVClassifier.pdf')


# In[229]:

SVClassifier.predict([1.0, 1.5])


# # Thank you for attention
# 
# Questions? [@Balzer82](https://twitter.com/balzer82)
