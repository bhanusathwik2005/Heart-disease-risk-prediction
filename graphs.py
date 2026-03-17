import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/heart_cleaned.csv")

# Graph 1: Disease Distribution
plt.figure()
data["target"].value_counts().plot(kind="bar")
plt.title("Heart Disease Distribution")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Count")
plt.savefig("static/disease_distribution.png")
plt.close()

# Graph 2: Age vs Disease
plt.figure()
plt.scatter(data["age"], data["target"])
plt.title("Age vs Heart Disease")
plt.xlabel("Age")
plt.ylabel("Disease")
plt.savefig("static/age_vs_disease.png")
plt.close()

print("Graphs generated successfully")
