import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#######################################################################################################################
# Importing Forex Data from 2015 until 2019
data = pd.read_csv (r'C:\Users\Vasco\Documents\Ms Data Science\1Semester\3Research Project 1\FX_Data_Russian\EURUSD_2015_2019_Volume_Data.txt',dtype=str)
data['DATETIME'] = data['DATE']+' '+data['TIME']
# data['DATETIME'] = data['DATE'].map(str)+' '+data['TIME'].map(str)
# data['<DATETIME>'] = data['<DATE>'].str[0:4]+'/'+data['<DATE>'].str[4:6]+'/'+data['<DATE>'].str[6:]+ ' ' + data['<TIME>'].str[0:2]+':'+data['<TIME>'].str[2:4]+':'+data['<TIME>'].str[4:6]
data = data.drop(['TICKER','PER','DATE','TIME'],axis=1)
# print(data.head(5))
# print(data.dtypes)

#######################################################################################################################
# Converting the data types
data['DATETIME'] = pd.to_datetime(data['DATETIME'],format='%Y%m%d %H%M%S')#Time in UTC
data['OPEN'] = pd.to_numeric(data['OPEN'],downcast='float')
data['HIGH'] = pd.to_numeric(data['HIGH'],downcast='float')
data['LOW'] = pd.to_numeric(data['LOW'],downcast='float')
data['CLOSE'] = pd.to_numeric(data['CLOSE'],downcast='float')
data['VOL'] = pd.to_numeric(data['VOL'],downcast='integer')
# print(data.head(5))
# print(data.dtypes)

#######################################################################################################################
# Basic Variable creation using Time
data['WEEKDAY'] = data.DATETIME.dt.weekday #monday=0
data['MONTH'] = data.DATETIME.dt.month #January=1
data['DAY'] = data.DATETIME.dt.day
data['HOUR'] = data.DATETIME.dt.hour

#######################################################################################################################
# Ploting some histograms
data['MONTH'].hist()
plt.title('Transactions per Month')
plt.show()
data['DAY'].hist()
plt.title('Transactions per Day of the Month')
plt.show()
data['WEEKDAY'].hist()
plt.title('Transactions per Weekday')
plt.show()
data['HOUR'].hist()
plt.title('Transactions per Hours of the Day (UTC TimeZone)')
plt.show()

#######################################################################################################################
plt.figure(1)
Timeline = plt.plot(data['DATETIME'],data['OPEN'])
plt.title('Currency pair USD/EUR 2015-2019')
plt.xlabel('Time')
plt.ylabel('Open price')
plt.show(Timeline)

#######################################################################################################################
# I have to make it weighted so the newer elements weight more that the older ones
data['open_MA24'] = data['OPEN'].rolling(window=24).mean()
data['open_MA50'] = data['OPEN'].rolling(window=50).mean()
data['open_MA100'] = data['OPEN'].rolling(window=100).mean()
data['open_MA700'] = data['OPEN'].rolling(window=700).mean()

plt.figure(2)
ax = plt.subplot()
ax.plot(data['DATETIME'],data['OPEN'],label='Original Data')
ax.plot(data['DATETIME'],data['open_MA24'],label='open_MA24')
ax.plot(data['DATETIME'],data['open_MA50'],label='open_MA50')
ax.plot(data['DATETIME'],data['open_MA100'],label='open_MA100')
ax.plot(data['DATETIME'],data['open_MA700'],label='open_MA700')
ax.legend()
plt.show()

#######################################################################################################################
#fourier transform to try to see if there is a constant frequency in the signal
n = len(data)
freq = np.fft.fftfreq(n)
fft_calc = np.fft.fft(data['OPEN'])

plt.figure(3)
plt.title('Fourier Transform')
plt.plot(freq,fft_calc)
plt.show()

#######################################################################################################################
#Detrending the timeseries (OPEN variable) using differencing:
def detrend(var_name):
    diff = [0]
    for i in range(1, len(data[var_name])):
        value = data[var_name][i] - data[var_name][i - 1]
        diff.append(value)
    return diff
#detrending all de variables:
data['open_detrend'] = detrend('OPEN')
data['close_detrend'] = detrend('CLOSE')
data['high_detrend'] = detrend('HIGH')
data['low_detrend'] = detrend('LOW')

#######################################################################################################################
plt.plot(data['DATETIME'],data['open_detrend'],label='Detrended Data')
plt.legend()
plt.show()
print(data.head(5))

#######################################################################################################################
#fourier transform to try to see if there is a constant frequency in the detrended signal
n = len(data)
freq = np.fft.fftfreq(n)
fft_calc = np.fft.fft(data['open_detrend'])

plt.figure(3)
plt.title('Fourier Transform')
plt.plot(freq,fft_calc)
plt.show()

