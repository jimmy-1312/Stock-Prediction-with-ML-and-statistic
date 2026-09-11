import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# close = pd.read_csv("./save/data/NVDA_2y.csv",index_col=0)["Close"]
# open = pd.read_csv("./save/data/NVDA_2y.csv",index_col=0)["Open"]


# def func(s):
#     return s[1]
# df = pd.read_csv("./save/features/alpha157_NVDA5y.csv",index_col=0)
# col_list = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
#             "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
#             "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
# ]
# plt.figure()
# plt.hist(df["RESI5"]**0.25,bins=100)
# # plt.figure()
# # plt.hist(df["KLOW"]**0.25,bins=100)
# plt.show()

# a = np.array([16777217,1],dtype=np.float32)
# print(np.float32(16777216)-np.float32(1))
# a = pd.Series([1,2,3,4,5,6],index=["1","2","3","4",'5','6'])
# b = pd.Series([2,3,4,5,6],index=["2","3","4","5","6"])
# print(a.rolling(window=5,center=True,closed="right").apply(lambda s:s[-1]/s[-2]-1,raw=True))
with open("finished_feature.txt", "w", encoding="utf-8") as f:
    f.write(f"2\n")
    f.write(f"1\n")