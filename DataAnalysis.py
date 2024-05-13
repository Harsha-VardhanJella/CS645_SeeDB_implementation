import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('adult.data', header=None, names=['Age', 'Workclass', 'Final Weight', 'Education', 'Education Number', 
                                                   'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 
                                                   'Capital Gain', 'Capital Loss', 'Hours per Week', 'Native Country', 'Income'])

numerical_features = ['Age', 'Final Weight', 'Education Number', 'Capital Gain', 'Capital Loss', 'Hours per Week']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, start=1):
    plt.subplot(3, 2, i)
    plt.hist(df[feature], bins=20, alpha=0.7, color='green')
    plt.title(f'Histogram of {feature}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
