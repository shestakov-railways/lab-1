import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

COMMENT = "\033[92m"
RESET = "\033[0m"

train=pd.read_csv(r'./input/train.csv')
df=train.copy()

# ### 1. BASICS

# output first 5 rows
print(f"{COMMENT}output first 5 rows{RESET}")
print(df.head())

# last 5 rows
print(f"{COMMENT}last 5 rows{RESET}")
print(df.tail())

# n_samples x n_features
print(f"{COMMENT}n_samples x n_features{RESET}")
print(df.shape)

#List of all the columns
print(f"{COMMENT}list all columns {RESET}")
print(df.columns)

# Rows index
print(f"{COMMENT}rows index {RESET}")
print(df.index)

# Values with their counts in a particular column
print(f"{COMMENT}Values with their counts in a particular column {RESET}")
print(df['Pclass'].value_counts())

# General description of dataset.
print(f"{COMMENT}General description of dataset {RESET}")
print(df.describe())

# ### 2. CREATING DATAFRAME

# empty data frame
df_empty=pd.DataFrame()
print(f"{COMMENT}empty data frame {RESET}")
print(df_empty.head())

# From dict
student_dict={'Name':['A','B','C'],'Age':[24,18,17],'Roll':[1,2,3]}
df_student=pd.DataFrame(student_dict).reset_index(drop=True) # without this adds an additional index column in df
print(df_student.head())

# ### 3. Treating null values
print(f"{COMMENT}3) Treating null values — preview head{RESET}")
print(df.head())

print(f"{COMMENT}Nulls per column (whole df){RESET}")
print(df.isnull().sum())

print(f"{COMMENT}Nulls in Age{RESET}")
print(df['Age'].isnull().sum())

print(f"{COMMENT}Imputing: Age -> mean age{RESET}")
df['Age'].fillna(df['Age'].mean(), inplace=True)
print(df['Age'].isnull().sum())

print(f"{COMMENT}Imputing: Sex -> mode (if any are null){RESET}")
if df['Sex'].isnull().any():
    df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)
print(df['Sex'].isnull().sum())

# ### 4. Modify / add new columns
print(f"{COMMENT}4) Modify / Add new column(s){RESET}")
print(df.head())

print(f"{COMMENT}Map Sex -> '0' for male, '1' for female{RESET}")
df['Sex'] = df['Sex'].map({'male': '0', 'female': '1'})
print(df.head())

print(f"{COMMENT}Split Name into last_name / first_name{RESET}")
df['last_name']  = df['Name'].apply(lambda x: x.split(',')[0])
df['first_name'] = df['Name'].apply(lambda x: ' '.join(x.split(',')[1:]).strip())
print(df[['Name', 'last_name', 'first_name']].head())

print(f"{COMMENT}Flag men in 3rd class -> Thrid&Men (1/0){RESET}")
df['Thrid&Men'] = df.apply(lambda row: int(row['Pclass'] == 3 and row['Sex'] == '0'), axis=1)
print(df[['Pclass', 'Sex', 'Thrid&Men']].head())

print(f"{COMMENT}Derive Age_group via custom function{RESET}")
def findAgeGroup(age):
    if age < 18:
        return 1
    elif age < 40:
        return 2
    elif age < 60:
        return 3
    else:
        return 4

df['Age_group'] = df['Age'].apply(findAgeGroup)
print(df[['Age', 'Age_group']].head())

# ### 5. Deleting columns
print(f"{COMMENT}5) Deleting columns (drop PassengerId){RESET}")
print(df.head())
df = df.drop(['PassengerId'], axis=1)
print(df.head())

# ### 6. Renaming columns
print(f"{COMMENT}6) Renaming columns{RESET}")
df = df.rename(columns={
    'Sex': 'Gender',
    'Name': 'Full Name',
    'last_name': 'Surname',
    'first_name': 'Name'
})
print(df.head())

# ### 7. Slicing dataframe
print(f"{COMMENT}7.i) Slicing DataFrame{RESET}")
print(df.head())

print(f"{COMMENT}All rows with Pclass == 3{RESET}")
df_third_class = df[df['Pclass'] == 3].reset_index(drop=True)
print(df_third_class.head())

print(f"{COMMENT}Females with Age > 60 (Gender == '1'){RESET}")
df_aged = df[(df['Age'] > 60) & (df['Gender'] == '1')]
print(df_aged[['Full Name', 'Gender', 'Age']].head())

print(f"{COMMENT}Select a few columns explicitly{RESET}")
df1 = df[['Age', 'Pclass', 'Gender']]
print(df1.head())

print(f"{COMMENT}Select numerical columns only{RESET}")
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
print(df_num.head())

print(f"{COMMENT}Categorical (object) columns{RESET}")
df_cat = df.select_dtypes(include=['object'])
print(df_cat.head())

# ### 7. Slicing using iloc and loc
print(f"{COMMENT}7.ii) iloc{RESET}")
print(df.head())

print(f"{COMMENT}First 100 rows, all columns (iloc){RESET}")
df_sub1 = df.iloc[0:100, :]
print(df_sub1.head())

print(f"{COMMENT}First 250 rows with specific columns by index (iloc){RESET}")
df_sub2 = df.iloc[:250, [1, 8]]
print(df_sub2.head())

print(f"{COMMENT}loc{RESET}")
print(df.head())

print(f"{COMMENT}First 500 rows (loc slice by labels){RESET}")
df_sub3 = df.loc[:500, :]
print(df_sub3.head())

print(f"{COMMENT}Gender and Age for Age > 50 (loc with condition){RESET}")
df_sub4 = df.loc[df['Age'] > 50, ['Gender', 'Age']]
print(df_sub4.head())

# ### 8. Adding a row
print(f"{COMMENT}8) Adding a row{RESET}")
print(df.head())

row = {'Age': 24, 'Full Name': 'Peter', 'Survived': 'Y'}
print(f"{COMMENT}Add row using concat{RESET}")
df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
print(df.tail())

print(f"{COMMENT}Add another row using loc{RESET}")
df.loc[len(df.index)] = row
print(df.tail())

# ### 9. Dropping rows
print(f"{COMMENT}9) Dropping last row{RESET}")
df = df.drop(df.index[-1], axis=0)
print(df.head())

# ### 10. Sorting
print(f"{COMMENT}10) Sort by Age (descending){RESET}")
df = df.sort_values(by=['Age'], ascending=False)
print(df.head())

# ### 11. Joins
print(f"{COMMENT}11) Joins: build two tiny frames{RESET}")
sno  = [i + 1 for i in range(100)]
marks = np.random.randint(100, size=100)
marks_df = pd.DataFrame({'Sno': sno, 'Marks': marks})
print(len(marks))
print(marks_df.head())

age = np.random.randint(100, size=100)
age_df = pd.DataFrame({'Sno': sno, 'Age': age})
print(len(marks))
print(age_df.head())

print(f"{COMMENT}Cross join{RESET}")
cross_join = pd.merge(marks_df, age_df, how='cross')
print(cross_join.shape)
print(cross_join.head())

print(f"{COMMENT}Inner join on Sno{RESET}")
inner_join = pd.merge(age_df, marks_df, how='inner', on='Sno')
print(inner_join.shape)
print(inner_join.head())

print(f"{COMMENT}Left/Right joins — extend age_df to create gaps{RESET}")
for s, a in [(101, 23), (102, 27), (104, 29), (103, 32), (105, 53)]:
    age_df.loc[len(age_df.index)] = {'Sno': s, 'Age': a}

left_join = pd.merge(age_df, marks_df, how='left', on='Sno')
print(left_join.shape)
print(left_join.tail())

right_join = pd.merge(marks_df, age_df, how='right', on='Sno')
print(right_join.shape)
print(right_join.tail())

print(f"{COMMENT}Full outer join — add an extra row to marks_df{RESET}")
marks_df.loc[len(marks_df.index)] = {'Sno': 106, 'Marks': 79}
out_join = pd.merge(marks_df, age_df, how='outer', on='Sno')
print(out_join.shape)
print(out_join.tail(10))

# ### 12. groupby
print(f"{COMMENT}12) GroupBy{RESET}")
print(df.head())

print(f"{COMMENT}Group by Pclass (from current df){RESET}")
groups = df.groupby(['Pclass'])
print(f"{COMMENT}Group (Pclass == 1){RESET}")
print(groups.get_group(1).head())

print(f"{COMMENT}Average Age per Pclass{RESET}")
df_grp1 = df.groupby(['Pclass'])
print(df_grp1['Age'].mean())

print(f"{COMMENT}Min Age per Pclass{RESET}")
print(df_grp1['Age'].min())

print(f"{COMMENT}Count of Age values per Pclass{RESET}")
print(df_grp1['Age'].count())

print(f"{COMMENT}Using agg(): reset to original train for clean examples{RESET}")
df = train.copy()
print(df.head())

print(f"{COMMENT}Average Age per Pclass via agg (skipna){RESET}")
df_grp2 = df.groupby(['Pclass']).agg({'Age': lambda s: s.mean()})
print(df_grp2.head())

print(f"{COMMENT}Min Age per Pclass via agg and rename{RESET}")
df_grp3 = df.groupby(['Pclass']).agg({'Age': 'min'}).rename(columns={'Age': 'Min Age'})
print(df_grp3.head())

print(f"{COMMENT}Join passenger names per class{RESET}")
df_grp4 = df.groupby(['Pclass']).agg({'Name': lambda s: ', '.join(s)})
print(df_grp4.head())

print(f"{COMMENT}Multiple aggregations on Age (max/min) -> MultiIndex columns{RESET}")
df_grp5 = df.groupby(['Pclass']).agg({'Age': ['max', 'min']})
print(df_grp5.head())

print(f"{COMMENT}Bonus: iterating rows{RESET}")
for idx, row in df.head(3).iterrows():
    print(row['Name'])
    pass

# Optional:
print(f"{COMMENT}Rough: array shape/preview{RESET}")
arr1 = np.random.randint(100, size=(100, 1))
arr2 = np.random.randint(100, size=(100,))
print(arr1[:5])
print(arr2[:20])