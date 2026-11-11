
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



df = yf.download("^N225", start="2000-01-01", auto_adjust=True)
print(df.tail())


df["Log_Return"]= np.log(df['Close'] / df['Close'].shift(1))
print(df)


plt.figure(figsize=(10, 10))
plt.plot(
    df.index,
    df["Log_Return"],
    linestyle="-",
    color="b",
    label="Daily Log Return",
)
plt.xlabel("Date")
plt.ylabel("Log return")
plt.legend()
plt.show()


print(df["Log_Return"].abs().nlargest(10))


r = df["Log_Return"]
print(r.mean(), r.std(), r.skew(), r.kurt() + 3, r.min(), r.max())



plt.figure(figsize=(10, 10))
plt.plot(
    df.index,
    df["Close"],
    linestyle="-",
    color="b",
    label="Daily Close",
)
plt.xlabel("Date")
plt.ylabel("Close")
plt.legend()
plt.show()


#compare how this looks like compared to gaussian 
plt.figure(figsize=(8, 5))
plt.hist(r, bins=100, density=True, alpha=0.6, label="Daily log returns")
x = np.linspace(r.min(), r.max(), 500)
plt.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color="red", label="Gaussian")
plt.yscale("log")
plt.ylim(bottom=1e-2)
plt.xlabel("Log return")
plt.ylabel("Density (log scale)")
plt.legend()
plt.show()



# Calculate if there is an autocorrelation due to time
# This can be done via ACF(Autocorrelation Function)
c_r = df['Log_Return'].dropna().tolist()

print(c_r)

def ACF(data, max_lag):
    n = len(data)
    mean = sum(data)/n
    variance = sum((x-mean)**2 for x in data)

    acf_value=[]
    for k in range(max_lag + 1):
        covariance = 0 
        for t in range(k,n):
            covariance += (data[t]- mean)*(data[t-k] - mean)

        acf_value.append(covariance/variance)

    return acf_value



r_ACF = print("AFC lag 0-10", ACF(c_r, max_lag=60))
 
r_sqrd = [x ** 2 for x in c_r]

r_sqrd_ACF=print("AFC lag 0-10", ACF(r_sqrd, max_lag=60))

rolling_vol = df['Log_Return'].rolling(window=60).std()

plt.figure(figsize=(10,10))
plt.plot(rolling_vol)
plt.xlabel('Date')
plt.ylabel('60-day standard deviation of daily log return')
plt.show()


max_lag = 60

acf_r  = ACF(c_r, max_lag)
acf_r2 = ACF(r_sqrd, max_lag)

n = len(c_r)
band = 1.96 / np.sqrt(n)          # rough "could be noise" band


fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

for ax, vals, title in zip(axes,
                           [acf_r, acf_r2],
                           ['Daily log returns', 'Squared daily log returns']):
    ax.bar(range(1, max_lag + 1), vals[1:])   # skip lag 0 (always 1)
    ax.axhline(band, linestyle='--', color='red')
    ax.axhline(-band, linestyle='--', color='red')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('Lag (trading days)')

axes[0].set_ylabel('Autocorrelation')
plt.tight_layout()
plt.show()



split_date = "2012-01-01"

train = df.loc[df.index < split_date]
test  = df.loc[df.index >= split_date]

print(len(train), len(test))
print(train.index[0], train.index[-1])
print(test.index[0], test.index[-1])

df.to_csv("data/n225.csv")

