import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


print("="*60)
print("LOADING DATA")
print("="*60)

#Load test data
test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
test_df = pd.read_csv(test_csv)

#Load predictions
try:
    predictions_df = pd.read_csv('results/test_predictions_lower_lr.csv')
    test_df['predicted'] = predictions_df['predicted'].values
    test_df['predicted_prob'] = predictions_df['predicted_prob'].values
    has_predictions = True
except FileNotFoundError:
    print("No predictions found. Run evaluate.py first with updated code.")
    has_predictions = False

#Check and standardise sex column
print("\n Sex column values:")
print(test_df['sex'].value_counts())

#Standardise sex column
if test_df['sex'].dtype == 'object':
    #If sex is 'Male'/'Female' strings
    test_df['sex'] = test_df['sex'].map({'Male': 'M', 'Female': 'F'})
else:
    #If sex is already 'M'/'F', keep as is
    pass

print("\n" + "="*60)
print("DEMOGRAPHIC PERFORMANCE ANALYSIS")
print("="*60)

#Age groups
test_df['age_group'] = pd.cut(test_df['age'],
                              bins=[0, 70, 80, 100],
                              labels=['<70', '70-80', '>80'])

if has_predictions:
    print("\n1. Performance by Age Group:")
    for group in ['<70', '70-80', '>80']:
        subset = test_df[test_df['age_group'] == group]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            cm = confusion_matrix(subset['label'], subset['predicted'])

            #Calculate sensitivity/specificity if there are both classes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                print(f"   {group:6s}: {acc*100:5.1f}% acc, {sens*100:5.1f}% sens, {spec*100:5.1f}% spec (n={len(subset):2d})")
            else:
                print(f"   {group:6s}: {acc*100:5.1f}% acc (n={len(subset):2d})")

    print("\n2. Performance by Sex:")
    for sex in test_df['sex'].unique():
        subset = test_df[test_df['sex'] == sex]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            cm = confusion_matrix(subset['label'], subset['predicted'])

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                print(f"   {sex:6s}: {acc*100:5.1f}% acc, {sens*100:5.1f}% sens, {spec*100:5.1f}% spec (n={len(subset):2d})")
            else:
                print(f"   {sex:6s}: {acc*100:5.1f}% acc (n={len(subset):2d})")

    print("\n3. Performance by Education Level:")
    for educ in sorted(test_df['educ'].dropna().unique()):
        subset = test_df[test_df['educ'] == educ]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            print(f"   Level {int(educ)}: {acc*100:5.1f}% acc (n={len(subset):2d})")

    #Error Analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    #Separate correct vs incorrect predictions
    correct_mask = test_df['label'] == test_df['predicted']
    incorrect_mask = ~correct_mask

    correct_df = test_df[correct_mask]
    incorrect_df = test_df[incorrect_mask]

    print(f"\nCorrectly classified: {len(correct_df)} ({len(correct_df)/len(test_df)*100:.1f}%)")
    print(f"Incorrectly classified: {len(incorrect_df)} ({len(incorrect_df)/len(test_df)*100:.1f}%)")

    #Compare characteristics
    print("\n📊 Age comparison:")
    print(f"   Correct predictions:   {correct_df['age'].mean():.1f} ± {correct_df['age'].std():.1f} years")
    print(f"   Incorrect predictions: {incorrect_df['age'].mean():.1f} ± {incorrect_df['age'].std():.1f} years")

    if 'mmse' in test_df.columns:
        print("\n📊 MMSE comparison:")
        print(f"   Correct predictions:   {correct_df['mmse'].mean():.1f} ± {correct_df['mmse'].std():.1f}")
        print(f"   Incorrect predictions: {incorrect_df['mmse'].mean():.1f} ± {incorrect_df['mmse'].std():.1f}")

    print("\n📊 Sex distribution:")
    print("   Correct predictions:")
    print(correct_df['sex'].value_counts())
    print("   Incorrect predictions:")
    print(incorrect_df['sex'].value_counts())

    print("\n📊 Diagnosis distribution:")
    print("   Correct predictions:")
    print(correct_df['diagnosis'].value_counts())
    print("   Incorrect predictions:")
    print(incorrect_df['diagnosis'].value_counts())

    #Confusion matrix breakdown
    print("\n" + "="*60)
    print("DETAILED CONFUSION MATRIX ANALYSIS")
    print("="*60)

    cm = confusion_matrix(test_df['label'], test_df['predicted'])

    #True Negatives (CN predicted as CN)
    tn_mask = (test_df['label'] == 0) & (test_df['predicted'] == 0)
    #False Positives (CN predicted as VeryMild)
    fp_mask = (test_df['label'] == 0) & (test_df['predicted'] == 1)
    #False Negatives (VeryMild predicted as CN)
    fn_mask = (test_df['label'] == 1) & (test_df['predicted'] == 0)
    #True Positives (VeryMild predicted as VeryMild)
    tp_mask = (test_df['label'] == 1) & (test_df['predicted'] == 1)

    print(f"\nTrue Negatives (CN → CN): {tn_mask.sum()}")
    if tn_mask.sum() > 0:
        print(f"   Age: {test_df[tn_mask]['age'].mean():.1f} ± {test_df[tn_mask]['age'].std():.1f}")

    print(f"\nFalse Positives (CN → VeryMild): {fp_mask.sum()}")
    if fp_mask.sum() > 0:
        print(f"   Age: {test_df[fp_mask]['age'].mean():.1f} ± {test_df[fp_mask]['age'].std():.1f}")
        if 'mmse' in test_df.columns:
            print(f"   MMSE: {test_df[fp_mask]['mmse'].mean():.1f} ± {test_df[fp_mask]['mmse'].std():.1f}")

    print(f"\nFalse Negatives (VeryMild → CN): {fn_mask.sum()}")
    if fn_mask.sum() > 0:
        print(f"   Age: {test_df[fn_mask]['age'].mean():.1f} ± {test_df[fn_mask]['age'].std():.1f}")
        if 'mmse' in test_df.columns:
            print(f"   MMSE: {test_df[fn_mask]['mmse'].mean():.1f} ± {test_df[fn_mask]['mmse'].std():.1f}")

    print(f"\nTrue Positives (VeryMild → VeryMild): {tp_mask.sum()}")
    if tp_mask.sum() > 0:
        print(f"   Age: {test_df[tp_mask]['age'].mean():.1f} ± {test_df[tp_mask]['age'].std():.1f}")

    #Create visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    #Age distribution by prediction correctness
    axes[0, 0].hist([correct_df['age'], incorrect_df['age']],
                     label=['Correct', 'Incorrect'], bins=15, alpha=0.7)
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Age Distribution: Correct vs Incorrect')
    axes[0, 0].legend()

    #Performance by age group
    age_results = []
    for group in ['<70', '70-80', '>80']:
        subset = test_df[test_df['age_group'] == group]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            age_results.append({'group': group, 'accuracy': acc})

    age_df = pd.DataFrame(age_results)
    axes[0, 1].bar(age_df['group'], age_df['accuracy'])
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Performance by Age Group')
    axes[0, 1].set_ylim([0, 1])

    #Performance by sex
    sex_results = []
    for sex in test_df['sex'].unique():
        subset = test_df[test_df['sex'] == sex]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            sex_results.append({'sex': sex, 'accuracy': acc})

    sex_df = pd.DataFrame(sex_results)
    axes[1, 0].bar(sex_df['sex'], sex_df['accuracy'])
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Performance by Sex')
    axes[1, 0].set_ylim([0, 1])

    #Confusion matrix by group
    group_cm = []
    for group in ['<70', '70-80', '>80']:
        subset = test_df[test_df['age_group'] == group]
        if len(subset) > 0:
            cm_group = confusion_matrix(subset['label'], subset['predicted'])
            if cm_group.shape == (2, 2):
                tn, fp, fn, tp = cm_group.ravel()
                group_cm.append({
                    'group': group,
                    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
                })

    if group_cm:
        cm_df = pd.DataFrame(group_cm).set_index('group')
        cm_df.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Confusion Matrix Breakdown by Age')
        axes[1, 1].legend(['TN', 'FP', 'FN', 'TP'])

    plt.tight_layout()
    plt.savefig('results/demographic_analysis_lower_lr.png', dpi=150, bbox_inches='tight')
    print("\n Demographic analysis plot saved to: results/demographic_analysis.png")

else:
    print("\nCannot perform analysis without predictions.")
    print("Please update evaluate.py to save predictions, then re-run it.")

print("\n" + "="*60)