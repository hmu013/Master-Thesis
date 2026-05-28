import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def analyze_forensics(csv_path, categories):
    df = pd.read_csv(csv_path)
    
    df['pred'] = (df['prob'] >= 0.5).astype(int)
    df['correct'] = (df['pred'] == df['label'])
    
    results = []
    valid_aurocs = []
    weights = []

    for cat in categories:
        group_df = df[df['key'].str.contains(cat, case=False, na=False)]
        
        if len(group_df) == 0:
            continue

        real_subset = group_df[group_df['label'] == 0]
        fake_subset = group_df[group_df['label'] == 1]

        acc_real = real_subset['correct'].mean() if not real_subset.empty else np.nan
        acc_fake = fake_subset['correct'].mean() if not fake_subset.empty else np.nan
        
        try:
            group_auroc = roc_auc_score(group_df['label'], group_df['prob'])
            valid_aurocs.append(group_auroc)
            weights.append(len(group_df))
        except ValueError:
            group_auroc = np.nan

        results.append({
            "Category": cat,
            "Total": len(group_df),
            "Overall Acc": group_df['correct'].mean(),
            "AUROC": group_auroc,
            "Real Acc": acc_real,
            "Fake Acc": acc_fake
        })

    summary = pd.DataFrame(results)
    
    macro_avg_auc = np.mean(valid_aurocs) if valid_aurocs else 0
    weighted_avg_auc = np.average(valid_aurocs, weights=weights) if valid_aurocs else 0

    print("\n" + "="*95)
    print(f"{'PERFORMANCE REPORT':^95}")
    print("="*95)
    print(summary.to_string(
        index=False, 
        justify='center',
        formatters={
            'Overall Acc': '{:,.2%}'.format,
            'AUROC': lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A",
            'Real Acc': '{:,.2%}'.format,
            'Fake Acc': '{:,.2%}'.format
        }
    ))
    print("-" * 95)
    print(f"Macro-Average AUROC (Mean of Groups):      {macro_avg_auc:.4f}")
    print(f"Weighted-Average AUROC (Size Adjusted):    {weighted_avg_auc:.4f}")
    print("="* 95)
    
    return summary

if __name__ == "__main__":
    RESULTS_CSV = "Master-Theisis-Working-Repo/results/preds_epoch_18.csv"
    
    SUBSETS = ["dalle3", "kandinsky2","kandinsky3","midjourney","pixart1024",
               "playground-25","sdxl","sdxl_dpo","segmoe","ssd1b","stable-cascade",
               "vega","wurstchen2"]
    
    analyze_forensics(RESULTS_CSV, SUBSETS)