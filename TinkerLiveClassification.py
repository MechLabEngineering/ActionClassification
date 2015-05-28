
# coding: utf-8

# # Real Time Activity Classification
# 
# We'll use a [Tinkerforge IMU](http://www.tinkerforge.com/en/doc/Hardware/Bricks/IMU_Brick.html)
# 
# ![IMU](http://www.tinkerforge.com/en/doc/_images/Bricks/brick_imu_tilted_front_350.jpg)
# 
# ### Connection settings

# In[31]:

HOST = "localhost"
PORT = 4223
UID = "6xDXqN" # Change to your UID


# In[32]:

from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_imu import IMU


# ### Load the Classifier

# In[33]:

import pickle
with open('SVClassifier.pkl', 'rb') as fid:
    classifier = pickle.load(fid)

print('Classifier loaded.')


# In[34]:

import numpy as np


# ## Feature Calculation

# In[35]:

def fft_amplitude_peak(s):
    
    # don't forget the windowing to get rid of the leakage effect
    hann = np.hanning(len(s))
    
    # do the FFT with Hanning Window
    Yhann = np.fft.fft(hann*s)
    
    N = len(Yhann)/2+1
    Y = 2.0*np.abs(Yhann[:N])/N # right half is enough info
    
    return np.max(Y) # just return the maximum peak amplitude
    #return Y # return the full spectrum
    #return np.max(Y) / np.mean(Y) # return periodicity


# In[36]:

def accmaxmindiff(ax,ay,az):
    absacc = np.sqrt(ax**2 + ay**2 + az**2)
    return np.max(absacc)-np.min(absacc)


# In[37]:

def classify(sensordata):
    #print signal
    ax = np.array(sensordata)[...,0]
    ay = np.array(sensordata)[...,1]
    az = np.array(sensordata)[...,2]
    rollrate = np.array(sensordata)[...,3]
    pitchrate = np.array(sensordata)[...,4]
    yawrate = np.array(sensordata)[...,5]
    
    acc = accmaxmindiff(ax,ay,az)
    fft = fft_amplitude_peak(yawrate)

    activity = classifier.predict([acc, fft])

    print('%s' % (activity))


# In[38]:

def collect(ax, ay, az, rollrate, pitchrate, yawrate, temp, signals=[]):
    
    signals.append([ax, ay, az, rollrate, pitchrate, yawrate, temp])
    
    #print len(signals)
    ws = 1.0 # windowsize in seconds
    sp = 10.0 # sample period of sensor in milliseconds (see Callback of Tinkerforge IMU)
    
    if len(signals)>(ws/(sp/1000)):
        classify(signals) # send everything to classifier
        
        del signals[:] # clear signal vector


# In[ ]:




# In[39]:

def cb_imudynamic(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z, ang_x, ang_y, ang_z, temp):
    '''
    Gibt die kalibrierten Beschleunigungen des Beschleunigungsmessers für die X, Y und Z-Achse in mG zurück (G/1000, 1G = 9.80605m/s²).
    '''
    ax = acc_x/1000.0
    ay = acc_y/1000.0
    az = acc_z/1000.0

    '''
    Gibt die kalibrierten Winkelgeschwindigkeiten des Gyroskops für die X, Y und Z-Achse in °/14,375s zurück. (Um den Wert in °/s zu erhalten ist es notwendig durch 14,375 zu teilen)
    '''
    rollrate = ang_x/14.375*3.14/180.0
    pitchrate= ang_y/14.375*3.14/180.0
    yawrate =  ang_z/14.375*3.14/180.0

    temp = temp/100.0
    
    
    collect(ax, ay, az, rollrate, pitchrate, yawrate, temp)


# In[ ]:




# In[40]:

if __name__ == "__main__":
    
    ipcon = IPConnection() # Create IP connection
    imu = IMU(UID, ipcon) # Create device object

    ipcon.connect(HOST, PORT) # Connect to brickd
    # Don't use device before ipcon is connected
    print('IMU connected')
    
    # Register callback
    imu.set_all_data_period(10) #10ms
    imu.register_callback(imu.CALLBACK_ALL_DATA, cb_imudynamic)

    raw_input('Press key to exit\n') # Use input() in Python 3
    ipcon.disconnect()


# In[ ]:




# In[ ]:



