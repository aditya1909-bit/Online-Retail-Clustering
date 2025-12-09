#Aditya Dutta & Ege Gürsel

# Load cleaned transaction data and limit to the top N most frequent items
# to reduce dimensionality and sparsity for market basket analysis.

import pandas as pd
import numpy as np
import itertools

INPUT_FILE = 'cleaned_online_retail.csv'
MIN_SUPPORT = 0.01
MIN_LIFT = 1
TOP_ITEMS_LIMIT = 300

print(f"Loading cleaned data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

print(f"Filtering top {TOP_ITEMS_LIMIT} items by frequency...")
item_counts = df['Description'].value_counts()
top_items = item_counts.head(TOP_ITEMS_LIMIT).index
df_top = df[df['Description'].isin(top_items)]

# Build a basket matrix with one row per invoice and one column per item;
# entries are quantities, which will later be converted to binary presence/absence.
print("Binary Basket Creation...")
basket = df_top.pivot_table(index='Invoice', columns='Description', values='Quantity', aggfunc=np.sum).fillna(0)

# Convert the basket to a binary NumPy matrix for efficient support counting.
basket_binary = (basket > 0).astype(int)

matrix = basket_binary.values
item_names = basket_binary.columns.tolist()
n_transactions = matrix.shape[0]

print(f"Matrix shape: {matrix.shape}, Number of transactions: {n_transactions}")

# Use the Apriori algorithm to iteratively discover frequent itemsets
# that satisfy the minimum support threshold.
print("Generating Frequent Itemsets using Apriori...")

frequent_itemsets = []

# Step 1 of Apriori: compute support for all single items and keep those
# whose support exceeds the minimum support threshold.
print("Generating 1-itemsets...")
support_counts = np.sum(matrix, axis=0)
support_values = support_counts / n_transactions

L1 = [tuple([i]) for i in range(len(item_names)) if support_values[i] >= MIN_SUPPORT]

for item_idx in L1:
    frequent_itemsets.append({
        'itemset': item_idx,
        'support': support_values[item_idx[0]]
        })

print(f"Found {len(L1)} frequent 1-itemsets.")

k = 2
current_L = L1

while len(current_L) > 0:
    print(f"Generating {k}-itemsets...")
    
    # Generate candidate k-itemsets by combining item indices that appear
    # in the current frequent (k-1)-itemsets.
    unique_items_in_L = sorted(list(set(idx for itemset in current_L for idx in itemset)))
    candidates = list(itertools.combinations(unique_items_in_L, k))
    
    L_k = []
    
    # For each candidate itemset, count in how many transactions all items appear
    # together and compute its support.
    for cand in candidates:
        cand_indices = list(cand)
        
        count = np.sum(matrix[:, cand_indices].all(axis=1))
        support = count / n_transactions
        
        # Keep only candidates that meet the minimum support to form the next
        # level of frequent itemsets.
        if support >= MIN_SUPPORT:
            L_k.append(cand)
            frequent_itemsets.append({
                'itemset': cand,
                'support': support
                })
    
    # Keep only candidates that meet the minimum support to form the next
    # level of frequent itemsets.
    print(f"Found {len(L_k)} frequent {k}-itemsets.")
    current_L = L_k
    k += 1
    
    if k > 5:
        break

print(f"Total frequent itemsets found: {len(frequent_itemsets)}")

# After mining frequent itemsets, generate association rules and evaluate
# them using support, confidence, and lift.
print("Generating Association Rules...")
rules = []

def get_support(indicies):
    # Retrieve the support value for a given itemset (specified by indices) 
    # from the list of precomputed frequent itemsets.
    for f in frequent_itemsets:
        if set(f['itemset']) == set(indicies):
            return f['support']
    return 0

# For each frequent itemset, enumerate all possible non-empty proper subsets
# as antecedents and form rules antecedent -> consequent.
for item_Data in frequent_itemsets:
    itemset = item_Data['itemset']
    support_ac = item_Data['support']
    
    # Only itemsets of size greater than 1 can produce association rules.
    if len(itemset) > 1:
        for i in range(1, len(itemset)):
            for subset in itertools.combinations(itemset, i):
                subset = tuple(sorted(subset))
                remaining = tuple(sorted(set(itemset) - set(subset)))
                
                support_a = get_support(subset)
                support_b = get_support(remaining)
                
                # Compute confidence and lift for the candidate rule only if
                # both antecedent and consequent supports are non-zero.
                if support_a > 0 and support_b > 0:
                    confidence = support_ac / support_a
                    lift = support_ac / (support_a * support_b)
                    
                    if lift >= MIN_LIFT:
                        ant_names = [item_names[i] for i in subset]
                        cons_names = [item_names[i] for i in remaining]
                        
                        rules.append({
                            'antecedent': ", ".join(ant_names),
                            'consequent': ", ".join(cons_names),
                            'support': round(support_ac, 4),
                            'confidence': round(confidence, 4),
                            'lift': round(lift, 4)
                            })

# Convert the list of generated rules into a DataFrame, sort by lift,
# and optionally save the top rules to CSV.
rules_df = pd.DataFrame(rules)

if not rules_df.empty:
    rules_df = rules_df.sort_values(by='lift', ascending=False)
    print(f"Top 10 Association Rules by Lift:")
    print(rules_df.head(10).to_string(index=False))
    rules_df.to_csv('association_rules.csv', index=False)
    print("Association rules saved to association_rules.csv")
else:
    print("No association rules found.")