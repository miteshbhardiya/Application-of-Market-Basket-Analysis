#first we import all required libraries csv, pandas, and some functions from mlxtend
import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#now import the raw csv file and name it which is to be analysed
filename = "D:\\bsc ty project\\sample work\\transactions1.csv"
data = []

#we add all non-empty cells into data i.e list
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        non_empty_cells = [cell.strip() for cell in row if cell.strip()]
        if non_empty_cells:
            data.append(non_empty_cells)

#we further clean the data by removing duplicates from same transactions by converting list into set
for i in range(len(data)):
    data[i] = list(set(data[i]))

#we then use trasactionencoder to make a dataframe with boolean values
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head(5)

#we take help of inbuilt apriori algorithm to do analysis and store the output in another csv file
frequent_itemsets = apriori(df, min_support=0.02, max_len = 2,  use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
rules["consequents"] = rules["consequents"].apply(lambda x: list(x))
rules1 = rules.sort_values(by=['lift'], ascending=False)
rules1.to_csv('association_rule1.csv', index = False)