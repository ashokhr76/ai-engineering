import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'Employee_ID': [10, 20, 15, 25, 30],
    'Gender': ['M', 'F', 'F', 'M', 'F'],
    'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

categorical_columns = df.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_df], axis=1)

print("\nDataFrame after One-Hot Encoding:")
print(df)
