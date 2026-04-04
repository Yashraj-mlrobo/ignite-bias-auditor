import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv("data/adult_income.csv", header=None, names=columns, skipinitialspace=True)
df_test = pd.read_csv("data/adult_test.csv", header=None, names=columns, skipinitialspace=True, skiprows=1)
compas = pd.read_csv("data/compas.csv")

print("=== ADULT TRAIN ===")
print(df.isnull().sum())
print(f"\n'?' count in workclass: {(df['workclass'] == '?').sum()}")

print("\n=== ADULT TEST ===")
print(df_test.isnull().sum())

print("\n=== COMPAS ===")
print(compas.isnull().sum())