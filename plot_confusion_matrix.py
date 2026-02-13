# from sklearn import confusion_matrix
import pandas as pd

df = pd.read_csv('human_baseline_results.csv', index_col=0)

for i in range(1, 4):
    df_confusion = pd.crosstab(df['class'], df[f'human_{i}'], rownames=['Actual'], colnames=['Predicted'], margins=True)
    accuracy = (df['class'] == df[f'human_{i}']).mean()
    print(f"Confusion Matrix for human_{i}:")
    print(df_confusion)
    print(f"Accuracy for human_{i}: {accuracy}\n")

# combine all human predictions by stacking each human's (class, prediction) as rows
df_combined = pd.concat([
    df[['class', 'human_1']].rename(columns={'human_1': 'human'}),
    df[['class', 'human_2']].rename(columns={'human_2': 'human'}),
    df[['class', 'human_3']].rename(columns={'human_3': 'human'}),
], axis=0, ignore_index=True)
print(len(df_combined))

df_confusion_combined = pd.crosstab(df_combined['class'], df_combined['human'], rownames=['Actual'], colnames=['Predicted'], margins=True)
accuracy_combined = (df_combined['class'] == df_combined['human']).mean()
print("Confusion Matrix for combined human predictions:")
print(df_confusion_combined)
print(f"Accuracy for combined human predictions: {accuracy_combined}")