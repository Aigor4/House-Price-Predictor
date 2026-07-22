import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from etl import extract, transform
import pandas as pd
import numpy as np

dirty = extract("../data/train.csv")
df = transform(dirty)

df = pd.get_dummies(df)

y = df["SalePrice"]
y = np.log1p(y)
columns = ["SalePrice", "Id"]
X = df.drop(columns, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "random_forest": RandomForestRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
}

best_r2 = -float("inf")
best_model = None

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} r2_score: {r2:.2f} mean_squared_error: {mean_squared_error(y_test, y_pred):.2f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

Features = X.columns

joblib.dump(best_model, "../models/model.pkl")
joblib.dump(Features, "../models/feature_columns.pkl")