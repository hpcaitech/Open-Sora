import pandas as pd
from matplotlib import pyplot as plt


musa1 = pd.read_csv("./loss_curve/musa_loss_curve_2024-07-16 16:53:10.249999.csv", index_col=None)['0'].tolist()
musa2 = pd.read_csv("./loss_curve/musa_loss_curve_2024-07-19 13:06:58.027162.csv", index_col=None)['0'].tolist()

nv1 = pd.read_csv("./loss_curve/nv_loss_curve_2024-07-18_23_42_02_517923.csv", index_col=None)['0'].tolist()
nv2 = pd.read_csv("./loss_curve/nv_loss_curve_2024-07-19_00_57_10_834244.csv", index_col=None)['0'].tolist()
nv3 = pd.read_csv("./loss_curve/nv_loss_curve_2024-07-19_02_01_43_316622.csv", index_col=None)['0'].tolist()
nv4 = pd.read_csv("./loss_curve/nv_loss_curve_2024-07-19_03_18_06_423878.csv", index_col=None)['0'].tolist()
count = 0
for i in range(len(musa1)):
    y2 = (musa1[i] + musa2[i]) / 2
    y1 = (nv1[i] + nv2[i] + nv3[i] + nv4[i]) / 4
    print(abs((y2 - y1)/y1))
    if abs((y2 - y1)/y1) > 0.02:
        count += 1
print(f"count {count}")
