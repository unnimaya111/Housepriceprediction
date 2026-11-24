import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_house_prices.csv")

print(df['Price_INR'].describe())

plt.figure(figsize=(8, 5))
sns.histplot(df['Price_INR'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (INR)')
plt.ylabel('Number of Houses')
plt.show()

print(df.groupby('Location')['Price_INR'].mean())

df.groupby('Location')['Price_INR'].mean().plot(kind='bar')

plt.title('Average House Price by Location')
plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(df['Area_sqft'], df['Price_INR'])

plt.title('House Area vs Price')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (INR)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(x=df['Price_INR'])

plt.title('House Price Distribution and Outliers')
plt.xlabel('Price (INR)')
plt.xlim(left=0)

plt.tight_layout()
plt.show()