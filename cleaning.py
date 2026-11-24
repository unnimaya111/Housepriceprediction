
import pandas as pd

df = pd.read_csv("03_house_price_prediction.csv")

print("Original Dataset Information:")
df.info()
print(df.head())
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Clean categorical text
df['Location'] = df['Location'].str.strip().str.title()
df['Property_Type'] = df['Property_Type'].str.strip().str.title()
df['Furnishing'] = df['Furnishing'].str.strip().str.title()

# Correct inconsistent category names
df['Location'] = df['Location'].replace({
    'Bangalore': 'Bengaluru',
    'Trivandrum': 'Thiruvananthapuram'
})

df['Furnishing'] = df['Furnishing'].replace({
    'Semi Furnished': 'Semi-Furnished',
    'Un Furnished': 'Unfurnished'
})

# Check categorical values
columns = ['Location', 'Property_Type', 'Furnishing']

for col in columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())
    print(df[col].value_counts())

# Replace zero area values with missing values
df.loc[df['Area_sqft'] == 0, 'Area_sqft'] = pd.NA

# Check negative values
numeric_columns = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Age_Years',
                   'Floor', 'Parking_Spaces', 'Construction_Quality', 'Price_INR']

print("\nNegative Values:")
print(df[numeric_columns].lt(0).sum())

# Fill missing numerical values with median
numeric_columns = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Age_Years',
                   'Floor', 'Parking_Spaces', 'Construction_Quality']

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
categorical_columns = ['Location', 'Property_Type', 'Furnishing']

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("data/cleaned_house_prices.csv", index=False)

# Final checks
print("\nCleaning Completed!")
print("Cleaned Dataset Shape:", df.shape)
print("Remaining Duplicates:", df.duplicated().sum())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
