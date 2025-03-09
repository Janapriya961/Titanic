import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "train.csv"
df = pd.read_csv(file_path)
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values in each column:")
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)
print("\nMissing values after handling:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())

sns.countplot( x = "Survived", data = df, palette = "viridis", legend = False)
plt.title("Survival Count")
plt.xlabel("Survived(0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

sns.countplot(x = "Sex", hue = "Survived", data = df, palette = "muted")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

sns.countplot(x = "Pclass", hue = "Survived", data = df, palette = "coolwarm")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class(1 = First, 3 = Third)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize = (8, 5))
sns.histplot(df["Age"], bins = 30, kde = True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include = ['number']).corr(), annot = True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation Heatmap")
plt.show()








