import pandas as pd

data = pd.read_csv('../data/train.csv')
df = pd.DataFrame(data)
print(f"kształt: \n{df.shape}")
print("###############################################")
df.info()
print("###############################################")
print(f"braki: \n{df.isna().sum().sort_values(ascending=False)}")
print("###############################################")
print(f"min/max/mean/std targetu: \n{df['SalePrice'].describe()}")