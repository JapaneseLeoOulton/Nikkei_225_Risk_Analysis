import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/n225.csv", index_col=0, parse_dates=True)


r = df["Log_Return"]
table = pd.DataFrame({"y": r, "lag1": r.shift(1)}).dropna()
print(table.head())


train = table.loc[:"2011"]             # rows up to end of 2011
X_train = train[["lag1"]]              # keeps it 2D
y_train = train["y"]

model = LinearRegression()          
model.fit(X_train, y_train)                    

print(model.intercept_ ,model.coef_)                         


first_test = table.loc["2012":].index[0]          # first test date
x_new = table.loc[[first_test], ["lag1"]]         # double brackets again
print(model.predict(x_new))               

test_dates = table.loc["2012":].index
forecasts = {}

for t in test_dates:
    train = table.loc[:t].iloc[:-1]            
    model = LinearRegression()
    model.fit(train[["lag1"]], train["y"])     # refit on the longer slice
    x_new = table.loc[[t], ["lag1"]]           # just day t's input
    forecasts[t] = model.predict(x_new)[0]

print(len(forecasts))


results = pd.DataFrame({
    "actual": table.loc[test_dates, "y"],
    "ols": pd.Series(forecasts),
})
results["naive"] = 0.0

print(results.head())


sse_ols = np.sum((results['actual'] - results['ols']) ** 2)
sse_naive = np.sum((results['actual'] - results['naive']) ** 2)



r2_oos = 1 - (sse_ols / sse_naive)

print(r2_oos, sse_ols,sse_naive)

d = (results['actual'] - results['naive'])**2 - (results['actual'] - results['ols'])**2 #calculate what did OLS gained comapred t naive for each day 
print(d.mean() / (d.std() / np.sqrt(len(d))))   # rough t-statistic


d.cumsum().plot()
plt.title("Cumulative gain of OLS over naive (squared error)")
plt.ylabel("Cumulative reduction in squared error")
plt.show()

extreme = results['actual'].abs().nlargest(5).index
sub = results.drop(extreme)

sse_ols = np.sum((sub['actual'] - sub['ols']) ** 2)
sse_naive = np.sum((sub['actual'] - sub['naive']) ** 2)
r2_oos = 1 - (sse_ols / sse_naive)
print(r2_oos)

print(sub.head())


   ## now do it with up to 5 lags 


MAX_lag = 5 

table = pd.DataFrame(
    {"y": r, **{f"lag{k}": r.shift(k) for k in range (1, MAX_lag + 1)}}
).dropna()

actual = table.loc[test_dates, "y"]


def walk_forward(table, p, test_dates):
    cols =[f"lag{k}" for k in range(1, p+1)]
    pred = {}
    for t in test_dates:
        train = table.loc[:t].iloc[:-1] # leak prevented
        model = LinearRegression().fit(train[cols], train["y"])
        pred[t] = model.predict(table.loc[[t], cols])[0]
    return pd.Series(pred)

def expanding_mean(table, test_dates): # Naive forecast average training return up to day before t
    return pd.Series({t: table.loc[:t].iloc[:-1]["y"].mean() for t in test_dates})

def score(actual, forcast, naive=0.0):
    err_model = (actual-forcast) ** 2
    err_naive = (actual - naive) ** 2
    dif = err_naive - err_model
    r2 = 1- err_model.sum()/ err_naive.sum()
    t_stat = dif.mean()/ (dif.std()/np.sqrt(len(dif))) # still approximate and assumptions flawed but still as refrence
    return r2, t_stat

    
mean_fc = expanding_mean(table, test_dates)

rows = []
for p in range(1, MAX_lag+1):
    f = walk_forward(table, p, test_dates)
    r2_zero , t_zero = score(actual, f)
    r2_mean, t_mean = score(actual, f, naive=mean_fc)
    rows.append((p, r2_zero, t_zero, r2_mean, t_mean))

summary = pd.DataFrame(
    rows, columns=["p", "r2_vs_zero", "t_vs_zero", "r2_vs_mean", "t_vs_mean"]
)
print(summary.round(5))