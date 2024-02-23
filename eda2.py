import pandas as pd
import matplotlib.pyplot as plt

cbcl = pd.read_csv('cbcl_1_5-2023-07-21.csv')
iq = pd.read_csv("iq-2023-07-21.csv")

print(cbcl.shape)
print(iq.shape)

intersect = pd.merge(cbcl, iq, on="subject_sp_id", how="inner")
print(intersect.shape)

# counts of various iq tests
count_table = intersect['fsiq'].value_counts().reset_index()
count_table.columns = ['fsiq', 'count']
print(count_table)

print(intersect['fsiq_score'].mean())

# Plotting the distribution
plt.figure(figsize=(10, 6))
plt.hist(intersect['fsiq_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of fsiq_score')
plt.xlabel('fsiq_score Values')
plt.ylabel('Frequency')
plt.show()

plt.plot(intersect['fsiq_score'], marker='o', linestyle='-', color='skyblue')

