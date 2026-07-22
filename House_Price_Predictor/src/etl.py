import pandas as pd
import numpy as np

def extract(file):
    df = pd.read_csv(file)
    return df

def transform(df):

    none_fill_cols = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
                      "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
                      "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

    zero_fill_cols = ["MasVnrArea", "GarageYrBlt"]

    df[none_fill_cols] = df[none_fill_cols].fillna("None")
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    return df

def load(file, file_path):
    file.to_csv(file_path, index=False)
    return file

def main():
    df = extract("../data/train.csv")
    transformed = transform(df)
    file_path = "../data/Clean_train.csv"
    clean = load(transformed, file_path)
    print(clean.shape)

if __name__ == "__main__":
    main()