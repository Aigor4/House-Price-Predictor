import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from etl import extract, transform
def main():
    df = extract("../data/train.csv")
    df = transform(df)
    df = pd.get_dummies(df)

    y = df["SalePrice"]
    y = np.log1p(y)
    columns = ["SalePrice", "Id"]
    X = df.drop(columns, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("../models/model.pkl")
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, xmin=0, xmax=1, linestyle="--")
    plt.xlabel("y_pred")
    plt.ylabel("residuals")
    plt.title("residuals plot")

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.title("hist of residuals")
    plt.xlabel("residuals")
    plt.ylabel("count")

    plt.show()


if __name__ == "__main__": main()