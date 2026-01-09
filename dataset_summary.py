import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata
df = pd.read_csv(r'C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\metadata.csv')

# Keep one scan per patient
df_patients = df.sort_values('scan_filename').groupby('patient_id').first().reset_index()

print("="*60)
print("DATASET SUMMARY - OASIS-1 Cohort")
print("="*60)

print(f"\nTotal unique patients: {len(df_patients)}")
print(f"Total scans: {len(df)}")

print(f"\nClass Distribution:")
class_dist = df_patients['diagnosis'].value_counts()
print(class_dist)
print(f"\nPercentages:")
print(df_patients['diagnosis'].value_counts(normalize=True) * 100)

print(f"\nAge Statistics by Group:")
print(df_patients.groupby('diagnosis')['age'].describe())

print(f"\nSex Distribution:")
print(pd.crosstab(df_patients['diagnosis'], df_patients['sex'], margins=True))

print(f"\nEducation Statistics:")
print(df_patients.groupby('diagnosis')['educ'].describe())

print(f"\nMMSE Statistics:")
print(df_patients.groupby('diagnosis')['mmse'].describe())

# Save demographic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
df_patients.boxplot(column='age', by='diagnosis', ax=axes[0,0])
axes[0,0].set_title('Age Distribution by Diagnosis')
axes[0,0].set_xlabel('Diagnosis')
axes[0,0].set_ylabel('Age (years)')

# Sex distribution
pd.crosstab(df_patients['diagnosis'], df_patients['sex']).plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Sex Distribution by Diagnosis')
axes[0,1].set_xlabel('Diagnosis')
axes[0,1].set_ylabel('Count')
axes[0,1].legend(title='Sex')

# Education distribution
df_patients.boxplot(column='educ', by='diagnosis', ax=axes[1,0])
axes[1,0].set_title('Education Distribution by Diagnosis')
axes[1,0].set_xlabel('Diagnosis')
axes[1,0].set_ylabel('Education (years)')

# MMSE distribution
df_patients.boxplot(column='mmse', by='diagnosis', ax=axes[1,1])
axes[1,1].set_title('MMSE Score Distribution by Diagnosis')
axes[1,1].set_xlabel('Diagnosis')
axes[1,1].set_ylabel('MMSE Score')

plt.tight_layout()
plt.savefig('results/demographics_summary.png', dpi=150)
print("\n✓ Demographics plots saved to: results/demographics_summary.png")

print("\n" + "="*60)