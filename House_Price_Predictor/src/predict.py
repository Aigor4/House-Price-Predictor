import joblib
import numpy as np
import pandas as pd

model = joblib.load("../models/model.pkl")
features_columns = joblib.load("../models/feature_columns.pkl")

jakosc = int(input("Podaj ocene jakosci wykonczenia domu: "))
pow = float(input("Podaj powierzchnie mieszkalna domu (w stopach kwadratowych): "))
rok = int(input("Podaj rok wykonczenia domu: "))
piw = float(input("Podaj powierzchnie piwnicy domu (w stopach kwadratowych): "))
cars = int(input("Podaj ilosc miejsc w garazu: "))
laz = int(input("Podaj ilosc lazienek: "))

row = {col: 0 for col in features_columns}

row["OverallQual"] = jakosc
row["GrLivArea"] = pow
row["YearBuilt"] = rok
row["TotalBsmtSF"] = piw
row["GarageCars"] = cars
row["FullBath"] = laz

df = pd.DataFrame([row])
df = df[features_columns]
wynik = model.predict(df)
wynik = np.expm1(wynik[0])
print(f"Przewidywana cena domu: {wynik}")

