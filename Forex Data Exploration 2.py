import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

########################################################################################################################

# Importing Forex Data from 2015 until 2019
data = pd.read_csv (r'C:\Users\Vasco\Documents\Ms Data Science\1Semester\3Research Project 1\FX_Data_Russian\EURUSD_2015_2019_Volume_Data.txt',dtype=str)
data['DATETIME'] = data['DATE']+' '+data['TIME']
# data['DATETIME'] = data['DATE'].map(str)+' '+data['TIME'].map(str)
# data['<DATETIME>'] = data['<DATE>'].str[0:4]+'/'+data['<DATE>'].str[4:6]+'/'+data['<DATE>'].str[6:]+ ' ' + data['<TIME>'].str[0:2]+':'+data['<TIME>'].str[2:4]+':'+data['<TIME>'].str[4:6]
data = data.drop(['TICKER','PER','DATE','TIME'],axis=1)
# print(data.head(5))
# print(data.dtypes)

########################################################################################################################

# Converting the data types
data['DATETIME'] = pd.to_datetime(data['DATETIME'],format='%Y%m%d %H%M%S')#Time in UTC
data['OPEN'] = pd.to_numeric(data['OPEN'],downcast='float')
data['HIGH'] = pd.to_numeric(data['HIGH'],downcast='float')
data['LOW'] = pd.to_numeric(data['LOW'],downcast='float')
data['CLOSE'] = pd.to_numeric(data['CLOSE'],downcast='float')
data['VOL'] = pd.to_numeric(data['VOL'],downcast='integer')
# print(data.head(5))
# print(data.dtypes)

########################################################################################################################

# Basic Variable creation using Time
data['WEEKDAY'] = data.DATETIME.dt.weekday #monday=0
data['MONTH'] = data.DATETIME.dt.month #January=1
data['DAY'] = data.DATETIME.dt.day
data['HOUR'] = data.DATETIME.dt.hour

# Ploting some histograms
plt.figure(1)
plt.subplot(2, 2, 1)
data['MONTH'].hist()
plt.title('Transactions per Month')
plt.subplot(2, 2, 2)
data['DAY'].hist()
plt.title('Transactions per Day of the Month')
plt.subplot(2, 2, 3)
data['WEEKDAY'].hist()
plt.title('Transactions per Weekday')
plt.subplot(2, 2, 4)
data['HOUR'].hist()
plt.title('Transactions per Hours of the Day (UTC TimeZone)')

########################################################################################################################

plt.figure(2)
Timeline = plt.plot(data['DATETIME'],data['OPEN'])
plt.title('Currency pair USD/EUR 2015-2019')
plt.xlabel('Time')
plt.ylabel('Open price')

########################################################################################################################

# I have to make it weighted so the newer elements weight more that the older ones

# Defining a function to create moving averages for our variables
def MA(var_name,MAwindow):
    data[str(var_name+'_MA'+str(MAwindow))] = data[var_name].rolling(window=MAwindow).mean()
    return data[str(var_name+'_MA'+str(MAwindow))]

#Creating the variables
MA('OPEN',24)
MA('OPEN',50)
MA('OPEN',100)
MA('OPEN',700)
MA('OPEN',10000)

# MA('CLOSE',24)
# MA('CLOSE',50)
# MA('CLOSE',100)
# MA('CLOSE',700)
# MA('CLOSE',10000)

# MA('HIGH',24)
# MA('HIGH',50)
# MA('HIGH',100)
# MA('HIGH',700)
# MA('HIGH',10000)

# MA('LOW',24)
# MA('LOW',50)
# MA('LOW',100)
# MA('LOW',700)
# MA('LOW',10000)

plt.figure(3)
ax = plt.subplot()
ax.plot(data['DATETIME'],data['OPEN'],label='Original Data')
ax.plot(data['DATETIME'],data['OPEN_MA24'],label='OPEN_MA24')
ax.plot(data['DATETIME'],data['OPEN_MA50'],label='OPEN_MA50')
ax.plot(data['DATETIME'],data['OPEN_MA100'],label='OPEN_MA100')
ax.plot(data['DATETIME'],data['OPEN_MA700'],label='OPEN_MA700')
ax.plot(data['DATETIME'],data['OPEN_MA10000'],label='OPEN_MA10000')
ax.legend()
plt.title('Original data (Open) and Moving Averages')

########################################################################################################################

#fourier transform to try to see if there is a constant frequency in the signal
fft_calc = np.fft.fft(data['OPEN'])

n = len(data)
freq = np.fft.fftfreq(n)

plt.figure(4)
plt.title('Fourier Transform on Original Data')
plt.plot(freq[0:int(len(freq)/2)] , np.abs(fft_calc)[0:int(len(fft_calc)/2)]) # only plots half of the spectrum (positive)


########################################################################################################################

#Detrending the timeseries using differencing:
def detrend_diff(var_name):
    diff = [0]
    for i in range(1, len(data[var_name])):
        value = data[var_name][i] - data[var_name][i - 1]
        diff.append(value)
    return diff

#Detrending the timeseries using Moving average(window=10000):
def detrend_MA(var_name):
    diff = data[var_name] - data[str(var_name+'_MA10000')]
    return diff

#detrending all de variables (I think it's better if we use detrending with MA):
data['open_detrend'] = detrend_diff('OPEN')
# data['close_detrend'] = detrend_MA('CLOSE')
# data['high_detrend'] = detrend_MA('HIGH')
# data['low_detrend'] = detrend_MA('LOW')

########################################################################################################################

plt.figure(5)
plt.plot(data['DATETIME'],data['open_detrend'],label='Detrended Data')
plt.legend()
plt.title('Detrended Data')

print(data.head(5))

########################################################################################################################

# #Exeepriment of fft with a pure sine wave:
# #Creating a signal:
# Fs = 400 #Sampling frequency
# f = 50 #Hz
# sample = 100
# x = np.arange(sample)
# signal = np.sin(2 * np.pi * f * x / Fs)

# #Perform fourrier transform on the signal:
# fft_calc = np.fft.fft(signal)

# #Create the x axis interval to plot the fft:
# freq = np.fft.fftfreq(sample,1./Fs)

# #plotting the original signal and its fft:
# plt.figure(6)
# plt.title('Fourier Transform')
# # plt.plot(x,signal)
# plt.plot(freq[0:int(len(freq)/2)],np.abs(fft_calc)[0:int(len(fft_calc)/2)])
#

############################### END OF EXPERIMENT ###############################

#fourier transform to try to see if there is a constant frequency in the detrended signal

# print(np.count_nonzero(~np.isnan(data['open_detrend']))) #to check how many NAN values there are in the detrended data

signal=data['open_detrend'][9999:] #because the first 10000 are NaN due to the detrending with 10000MA
# signal=data['open_detrend'] #use this if detrended with detrend_diff

fft_calc = np.fft.fft(signal)

n = len(signal)
freq = np.fft.fftfreq(n,.01) #how should be the x axis?

plt.figure(7)
plt.title('Fourier Transform on Detrended Data')
# plt.plot(data['DATETIME'],data['open_detrend'])
plt.plot(freq[0:int(len(freq)/2)],np.abs(fft_calc)[0:int(len(fft_calc)/2)]) # only plots half of the spectrum (positive)

plt.show()
########################################################################################################################

#print(data['open_detrend'].size)
#print(freq.size)
#print(fft_calc.size)
