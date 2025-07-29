import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, col):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

def bivariate_analysis(df, x, y):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[x], y=df[y])
    plt.title(f"{y} by {x}")
    plt.show()