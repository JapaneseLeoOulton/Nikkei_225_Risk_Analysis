import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



df = yf.download("^N225", start="2000-01-01", auto_adjust=True)
df["Log_Return"] = np.log(df['Close'] / df['Close'].shift(1))

split_date = "2012-01-01"

train = df.loc[df.index < split_date]
test  = df.loc[df.index >= split_date]


print  (train.head())
len(df) == len(train)+len(test) 

smaple_mean = np.mean(train["Log_Return"])
sample_std = np.std(train["Log_Return"], ddof=1)

print(sample_std,smaple_mean)

VaR = -(smaple_mean + stats.norm.ppf(0.01)* sample_std)
print(VaR)

I = test["Log_Return"] < -VaR      
I.sum()                            # the number of breaches
I.mean()                           # breach rate, pi-hat




import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy



def kupiec_pof_test(x, N, alpha):
    pi_hat = x / N

    ll_null = (N - x) * np.log(1 - alpha) + x * np.log(alpha)

    ll_alt = xlogy(N - x, 1 - pi_hat) + xlogy(x, pi_hat)

    lr_uc = -2 * ll_null + 2 * ll_alt
    p_value = chi2.sf(lr_uc, df=1)

    return lr_uc, p_value


x = I.sum()
N = len(test)
lr_uc, p_value = kupiec_pof_test(x, N, 0.01)

print(f"x = {x}, N = {N}, breach rate = {x / N:.4%}")
print(f"LR_uc = {lr_uc:.3f}, p-value = {p_value:.3f}")

if p_value < 0.05:
    print("Reject H0: breach rate is inconsistent with 1%.")
else:
    print("Fail to reject H0: no evidence the breach rate differs from 1%.")



    

def christoffersen_ind_test(I):
    
    i = pd.Series(I).values

    t_0 = i[:-1]  # Day t
    t_1 = i[1:]  # Day t+1

    m00 = np.sum((t_0 == 0) & (t_1 == 0))
    m01 = np.sum((t_0 == 0) & (t_1 == 1))
    m10 = np.sum((t_0 == 1) & (t_1 == 0))
    m11 = np.sum((t_0 == 1) & (t_1 == 1))

    pi_01 = m01 / (m00 + m01) if (m00 + m01) > 0 else 0
    pi_11 = m11 / (m10 + m11) if (m10 + m11) > 0 else 0

    pi_overall = (m01 + m11) / (m00 + m01 + m10 + m11)

    L_null = ((1 - pi_overall) ** (m00 + m10)) * (pi_overall ** (m01 + m11))

    L_alt = (
        ((1 - pi_01) ** m00)
        * (pi_01**m01)
        * ((1 - pi_11) ** m10)
        * (pi_11**m11)
    )

    lr_ind = -2 * np.log(L_null) + 2 * np.log(L_alt)

    p_value = 1 - chi2.cdf(lr_ind, df=1)

    
    return {
        "Matrix": {"M00": m00, "M01": m01, "M10": m10, "M11": m11},
        "Probabilities": {"pi_01": pi_01, "pi_11": pi_11},
        "LR_ind": lr_ind,
        "p-value": p_value,
    }


christoffersen_ind_test(I)

breached_df = test[test["Log_Return"] < -VaR]
print(breached_df)

I_int = I.astype(int)
consec = (I_int == 1) & (I_int.shift(1) == 1)
print(test.index[consec])


df.index = pd.to_datetime(df.index)
df.index.name = "Date"


lr_cc = lr_uc + 6.7130745133115965  # this is lr_ind 
print(lr_cc)


p_val = 1- chi2.cdf(lr_cc, df = 2)
print(p_val)


pairs = consec | consec.shift(-1, fill_value=False)
print(test["Log_Return"][pairs])

test.index = pd.to_datetime(test.index)

r_test = test["Log_Return"]
breaches = r_test[I]                     # only the 40 breach rows


print(df.head(3))
print(type(df.index))

plt.figure(figsize=(12, 5))
plt.plot(r_test.index, r_test, linewidth=0.6, label="Daily log return")
plt.axhline(-VaR, color="red", linestyle="--", label="-VaR (99%)")
plt.scatter(breaches.index, breaches, color="red", s=20, zorder=3, label="Breaches (40)")
plt.legend()
plt.title("Gaussian 99% VaR: test period breaches")
plt.show()

print(type(test.index))