import pandas as pd

df = pd.read_csv("examples.csv")
for policy in df['policy_issue'].unique():
    for stance in df['stance'].unique():
        subset = df[(df['policy_issue'] == policy) & (df['stance'] == stance)]
        print(f"Policy: {policy}, Stance: {stance}")
        print(f"Count: {subset.statement.tolist()}")
        print()
