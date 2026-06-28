import pandas as pd

data = {'Colors': ['Red', 'Blue', 'Green', 'Red', 'Blue']}
df = pd.DataFrame(data)

df_encoder = pd.get_dummies(df,columns=['Colors'], prefix='Color')
print(df_encoder)


data_example = {
    'Employee_ID': [10, 20, 15, 25, 30],
    'Gender': ['M', 'F', 'F', 'M', 'F'],
    'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice']
}
df_example = pd.DataFrame(data_example)

df_example_encoded = pd.get_dummies(df_example, columns=['Gender', 'Remarks'], prefix=['Gender', 'Remark'])
print(df_example_encoded)
