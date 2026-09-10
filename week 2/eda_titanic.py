"""
Week 2 - Exploratory Data Analysis and Visualization
Dataset: Titanic passenger dataset cleaned during Week 1.

Purpose:
- Load the Week 1 cleaned dataset
- Perform descriptive EDA
- Explore survival patterns across demographic, class, fare and family variables
- Create annotated visualizations
- Compute correlations
- Produce reproducible summary tables

Run:
    python eda_titanic.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "titanic_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# 1. Initial inspection
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nDescriptive statistics:")
print(df[["Survived","Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone"]].describe())

# 2. Overall outcome
print("\nOverall survival rate:", df["Survived"].mean())

# 3. Grouped summaries
print("\nSurvival by sex:")
print(df.groupby("Sex")["Survived"].agg(["count","mean"]))

print("\nSurvival by class:")
print(df.groupby("Pclass")["Survived"].agg(["count","mean"]))

print("\nSurvival by age group:")
print(df.groupby("AgeGroup")["Survived"].agg(["count","mean"]))

print("\nSurvival by embarkation port:")
print(df.groupby("Embarked")["Survived"].agg(["count","mean"]))

# 4. Feature engineering used specifically for EDA
# Ticket group size: how many rows share each ticket?
df["TicketGroupSize"] = df.groupby("Ticket")["PassengerId"].transform("count")

# Fare per person is useful for identifying the effect of shared tickets.
df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

# 5. Visualizations
fig, ax = plt.subplots()
df["Survived"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Passenger Survival Distribution")
ax.set_xlabel("Survival Status (0 = No, 1 = Yes)")
ax.set_ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Survival by sex
sex_rate = df.groupby("Sex")["Survived"].mean()
ax = sex_rate.plot(kind="bar")
ax.set_title("Survival Rate by Sex")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Survival by passenger class
class_rate = df.groupby("Pclass")["Survived"].mean()
ax = class_rate.plot(kind="bar")
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Age distributions by outcome
fig, ax = plt.subplots()
ax.hist(df.loc[df["Survived"] == 0, "Age"], bins=20, alpha=.65, label="Did not survive")
ax.hist(df.loc[df["Survived"] == 1, "Age"], bins=20, alpha=.65, label="Survived")
ax.set_title("Age Distribution by Survival Outcome")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Passengers")
ax.legend()
plt.tight_layout()
plt.show()

# Fare distribution
fig, ax = plt.subplots()
ax.hist(df["Fare"], bins=35)
ax.set_title("Fare Distribution")
ax.set_xlabel("Fare")
ax.set_ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Family size
family_rate = df.groupby("FamilySize")["Survived"].mean()
ax = family_rate.loc[family_rate.index <= 6].plot(kind="line", marker="o")
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family Size")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Alone vs group
alone_rate = df.groupby("IsAlone")["Survived"].mean()
ax = alone_rate.plot(kind="bar")
ax.set_title("Survival Rate: Alone vs With Family")
ax.set_xlabel("IsAlone (0 = group, 1 = alone)")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Port
port_rate = df.groupby("Embarked")["Survived"].mean()
ax = port_rate.plot(kind="bar")
ax.set_title("Survival Rate by Port of Embarkation")
ax.set_xlabel("Embarked")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Correlation
corr_cols = ["Survived","Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
print("\nCorrelation matrix:")
print(df[corr_cols].corr())

# Interpretation reminder:
# Correlation identifies association, not causation.
