# Python: Basic Codes

# In this blog, I will share a basic set of Python commands and codes that will be helpful to start working with Python. These are easily available online, and this blog is just a small step to consolidate the important codes in one place. If someone really wants to learn Python, it is recommended to browse through coursera and edx.
# I had shared similar codes about data exploration in R in a previous blog. You can read it here.
# In this post, we will go through the following commands:

# Reading Data in Python
# List of columns in a dataset
# Appending two dataset
# Different joins in Python
# Where conditions and multiple where conditions
# Group by command
# Finding distinct values in a vector
# There are multiple modules available in python, and I have used Pandas for the basic operations.

# Importing the pandas module in the memory
import pandas as pd

# 1. Reading a .csv file in Python
Tran_Table1 = pd.read_csv('/Users/Desktop/Python_Dir/Tran_Table1.csv')
Tran_Table2 = pd.read_csv('/Users/Desktop/Python_Dir/Tran_Table2.csv')
Cust_Details = pd.read_csv('/Users/Desktop/Python_Dir/Cust_Details.csv')

# 2. The top 5 columns and column names for a data frame
Cust_Details.head()

# The list of column names for a data frame
Colname1 = list(Tran_Table1.columns.values)
Colname2 = list(Cust_Details.columns.values)

# 3. Merging (or appending) two data frames
frame1 = [Tran_Table1,Tran_Table2]
Total_Tran_Table = pd.concat(frame1)

# 4. Joins - Inner, Left, Right, Outer
Inner_Join = pd.merge(Tran_Table1, Cust_Details, how = 'inner', on = 'Cust_No')
Left_Join = pd.merge(Tran_Table1, Cust_Details, how = 'left', left_on = 'Cust_No', right_on = 'Cust_No')
Right_Join = pd.merge(Tran_Table1, Cust_Details, how = 'right', left_on = 'Cust_No', right_on = 'Cust_No')
Outer_Join = pd.merge(Tran_Table1, Cust_Details, how = 'outer', left_on = 'Cust_No', right_on = 'Cust_No')

# 5. 'Where' condition
Asian_Customer = Cust_Details[Cust_Details['Cust_Continent']=='Asia']
Asian_Customer

# Multiple 'Where' conditions
Asian_Male_Customer = Cust_Details[(Cust_Details['Cust_Continent']=='Asia') & (Cust_Details['Cust_Gender']=='Male')]
Asian_Male_Customer

# 6. Group By
Tran_Table1_Details = pd.merge(Tran_Table1, Cust_Details, how = 'inner', on = 'Cust_No')
Tran_Table1_Details.groupby('Cust_Continent')['X1'].sum()
Tran_Table1_Details.groupby(['Cust_Gender','Cust_Continent'])['Cust_No'].count()

# 7. Unique (distinct) values in a column
Unique_Continents = list(set(Cust_Details.Cust_Continent))
