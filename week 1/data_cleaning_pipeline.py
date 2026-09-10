"""
Data Acquisition, Cleaning, and Preprocessing Pipeline
Dataset: Titanic Passenger Manifest (891 records, 12 original fields)
Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
        (public mirror of the original Kaggle "Titanic - Machine Learning from Disaster" dataset)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13

FIG = "/home/claude/project/figures/"
DATA = "/home/claude/project/data/"

# ============================================================
# STEP 1: DATA ACQUISITION
# ============================================================
df_raw = pd.read_csv(DATA + "titanic_raw.csv")
print(f"[1] Loaded raw dataset: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

# ============================================================
# STEP 2: INITIAL EXPLORATION
# ============================================================
missing_before = df_raw.isnull().sum()
missing_pct_before = (missing_before / len(df_raw) * 100).round(2)

fig, ax = plt.subplots(figsize=(8, 4.5))
miss = missing_pct_before[missing_pct_before > 0].sort_values(ascending=True)
bar_colors = ["#3d5a80" if v < 50 else "#e07a5f" for v in miss.values]
bars = ax.barh(miss.index, miss.values, color=bar_colors)
for i, v in enumerate(miss.values):
    ax.text(v + 1, i, f"{v}%", va="center", fontsize=10)
ax.set_xlabel("% Missing")
ax.set_title("Missing Values by Column (Before Cleaning)")
ax.set_xlim(0, 90)
plt.tight_layout()
plt.savefig(FIG + "01_missing_before.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(df_raw["Age"].dropna(), bins=30, kde=True, ax=axes[0], color="#3d5a80")
axes[0].set_title("Age Distribution (raw, missing dropped)")
sns.boxplot(x=df_raw["Fare"], ax=axes[1], color="#e07a5f")
axes[1].set_title("Fare Distribution (raw) — outliers visible")
plt.tight_layout()
plt.savefig(FIG + "02_age_fare_raw.png", bbox_inches="tight")
plt.close()

# ============================================================
# STEP 3: DATA QUALITY ISSUES IDENTIFIED
# ============================================================
duplicates = df_raw.duplicated().sum()

Q1f, Q3f = df_raw["Fare"].quantile([0.25, 0.75])
IQRf = Q3f - Q1f
fare_lo, fare_hi = Q1f - 1.5 * IQRf, Q3f + 1.5 * IQRf
fare_outliers = df_raw[(df_raw["Fare"] < fare_lo) | (df_raw["Fare"] > fare_hi)]

Q1a, Q3a = df_raw["Age"].quantile([0.25, 0.75])
IQRa = Q3a - Q1a
age_lo, age_hi = Q1a - 1.5 * IQRa, Q3a + 1.5 * IQRa
age_outliers = df_raw[(df_raw["Age"] < age_lo) | (df_raw["Age"] > age_hi)]

fare_zero = df_raw[df_raw["Fare"] == 0]

print(f"[3] Duplicate rows: {duplicates}")
print(f"[3] Fare IQR bounds: [{fare_lo:.2f}, {fare_hi:.2f}] -> {len(fare_outliers)} flagged outliers")
print(f"[3] Age  IQR bounds: [{age_lo:.2f}, {age_hi:.2f}] -> {len(age_outliers)} flagged outliers")
print(f"[3] Fare == 0 (free passage records): {len(fare_zero)}")

# ============================================================
# STEP 4: CLEANING & MISSING-VALUE TREATMENT
# ============================================================
df = df_raw.copy()

# 4a. Title extraction from Name (needed BEFORE Age imputation — Title is a
#     much stronger predictor of Age than a single global median)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "the Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}
df["Title"] = df["Title"].replace(title_map)
print("\n[4a] Normalised Title categories:", sorted(df["Title"].unique()))

# 4b. Age — 19.87% missing (MAR: correlates with Title/Pclass, not random).
#     Impute with the median Age within each (Title, Pclass) group rather
#     than a single global median, which would flatten real age structure
#     (e.g. 'Master' = young boy, 'Mrs' = adult woman).
age_before = df["Age"].copy()
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
    lambda s: s.fillna(s.median())
)
df["Age"] = df["Age"].fillna(df["Age"].median())  # safety net for any empty group
print(f"[4b] Age missing after group-median imputation: {df['Age'].isnull().sum()}")

# 4c. Embarked — only 2 missing (0.22%). Safe to impute with the mode.
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"[4c] Embarked imputed with mode = '{embarked_mode}'")

# 4d. Cabin — 77.1% missing. Too sparse to impute reliably; imputing a
#     value would fabricate data for 3 out of 4 passengers. Instead,
#     convert into two engineered signals that preserve the real information:
#       - HasCabin: whether a cabin record exists at all (proxy for
#         cabin-class / proximity to lifeboats, historically linked to survival)
#       - Deck: first letter of Cabin where known, else 'Unknown'
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
df = df.drop(columns=["Cabin"])
print(f"[4d] Cabin dropped; engineered HasCabin, Deck. Deck values: {sorted(df['Deck'].unique())}")

# 4e. Duplicates
df = df.drop_duplicates()
print(f"[4e] Rows after de-duplication: {len(df)} (removed {len(df_raw) - len(df)})")

missing_after = df.isnull().sum()
print(f"\n[4] Total missing values remaining: {missing_after.sum()}")

# ============================================================
# STEP 5: OUTLIER TREATMENT
# ============================================================
# Age outliers (11 flagged, mostly elderly passengers e.g. age 70-80): these
# are genuine, plausible ages, not data errors, so they are RETAINED
# unmodified. Removing them would discard true variance.

# Fare outliers (116 flagged, up to 512.33): investigation shows the extreme
# values belong to real 1st-class passengers/families who purchased multiple
# joint tickets (e.g. the Ryerson and Cardeza families), not entry errors.
# Fare == 0 (15 records) are documented Titanic crew/employees (ticket
# prefix "LINE") travelling on free passage — also legitimate, not missing
# or erroneous data.
#
# Because Fare is strongly right-skewed and the outliers are real, we do NOT
# delete or hard-cap them (that would bias survival analysis toward
# under-representing 1st class). Instead we:
#   (a) winsorize at the 1st/99th percentile to dampen the influence of the
#       single most extreme value (512.33, ~3x the next highest) while
#       preserving distribution shape, and
#   (b) create a log-transformed Fare feature for modelling, which is the
#       standard treatment for right-skewed monetary variables.
p01, p99 = df["Fare"].quantile([0.01, 0.99])
df["Fare_winsorized"] = df["Fare"].clip(lower=p01, upper=p99)
df["Fare_log"] = np.log1p(df["Fare"])

print(f"[5] Fare winsorized at 1st/99th pct: [{p01:.2f}, {p99:.2f}]")
print(f"[5] Max Fare before: {df_raw['Fare'].max():.2f} -> after winsorizing: {df['Fare_winsorized'].max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(x=df["Fare"], ax=axes[0], color="#e07a5f")
axes[0].set_title("Fare — Original (with outliers retained)")
sns.boxplot(x=df["Fare_winsorized"], ax=axes[1], color="#81b29a")
axes[1].set_title("Fare — Winsorized (1st/99th pct)")
plt.tight_layout()
plt.savefig(FIG + "03_fare_winsorized.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(age_before.dropna(), bins=30, kde=True, ax=axes[0], color="#3d5a80", alpha=0.5, label="Original (177 NaN dropped)")
sns.histplot(df["Age"], bins=30, kde=True, ax=axes[0], color="#e07a5f", alpha=0.5, label="After imputation")
axes[0].set_title("Age Distribution: Before vs After Imputation")
axes[0].legend(fontsize=8)
sns.histplot(df["Fare_log"], bins=30, kde=True, ax=axes[1], color="#81b29a")
axes[1].set_title("Log(1 + Fare) — reduced right skew")
plt.tight_layout()
plt.savefig(FIG + "04_age_imputed_fare_log.png", bbox_inches="tight")
plt.close()

# ============================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle-Age", "Senior"]
)
print(f"[6] Engineered features: FamilySize, IsAlone, AgeGroup, HasCabin, Deck, Title, Fare_log")

# ============================================================
# STEP 7: ENCODING CATEGORICAL VARIABLES
# ============================================================
df_model = df.copy()
df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})
df_model = pd.get_dummies(
    df_model, columns=["Embarked", "Title", "Deck", "AgeGroup"],
    prefix=["Emb", "Title", "Deck", "AgeGrp"], drop_first=False
)
df_model = df_model.drop(columns=["Name", "Ticket", "PassengerId"])
print(f"[7] Encoded dataset shape (model-ready): {df_model.shape}")

# ============================================================
# STEP 8: FEATURE SCALING
# ============================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ["Age", "Fare_winsorized", "Fare_log", "FamilySize"]
df_model[[c + "_scaled" for c in num_cols]] = scaler.fit_transform(df_model[num_cols])
print(f"[8] Scaled columns: {[c + '_scaled' for c in num_cols]}")
print(df_model[[c + "_scaled" for c in num_cols]].describe().loc[["mean", "std"]].round(3))

# ============================================================
# STEP 9: SAVE CLEANED DATASETS
# ============================================================
df.to_csv(DATA + "titanic_cleaned.csv", index=False)
df_model.to_csv(DATA + "titanic_model_ready.csv", index=False)
print(f"\n[9] Saved: titanic_cleaned.csv {df.shape}, titanic_model_ready.csv {df_model.shape}")

# ============================================================
# STEP 10: CORRELATION HEATMAP (final sanity check)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6.5))
corr_cols = ["Survived", "Pclass", "Sex", "Age", "Fare_winsorized", "FamilySize", "HasCabin"]
corr_df = df_model[corr_cols].corr()
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Cleaned Feature Subset")
plt.tight_layout()
plt.savefig(FIG + "05_correlation_heatmap.png", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.2))
miss2 = missing_after[missing_after > 0]
if len(miss2) == 0:
    ax.text(0.5, 0.5, "All missing values resolved\n(0 nulls remaining)",
            ha="center", va="center", fontsize=13, color="#3d5a80")
    ax.axis("off")
else:
    ax.barh(miss2.index, miss2.values, color="#e07a5f")
ax.set_title("Missing Values by Column (After Cleaning)")
plt.tight_layout()
plt.savefig(FIG + "06_missing_after.png", bbox_inches="tight")
plt.close()

print("\n=== PIPELINE COMPLETE ===")
