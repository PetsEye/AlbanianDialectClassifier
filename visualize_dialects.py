import sys

import pandas as pd
import matplotlib.pyplot as plt

from common import CSV_PATH, LABELS, categorize_dialect

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    sys.exit(f"Error: '{CSV_PATH}' not found. Run the scraper first!")

df.columns = df.columns.str.strip()
if "status" in df.columns:
    df = df[df["status"] == "Success"]

df["Main_Dialect"] = df["dialect"].apply(categorize_dialect)
uncategorized = df["Main_Dialect"].isna().sum()
if uncategorized:
    print(f"Warning: {uncategorized} rows did not match a known dialect and are excluded.")
df = df.dropna(subset=["Main_Dialect"])

counts = df["Main_Dialect"].value_counts().reindex(LABELS).dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
counts.plot(kind='bar', color=colors, ax=ax1)
ax1.set_title('Total Count per Dialect', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Samples')
ax1.set_xlabel('Dialect Group')
ax1.tick_params(axis='x', rotation=0)

for i, v in enumerate(counts):
    ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

counts.plot(kind='pie', autopct='%1.1f%%', startangle=140,
            colors=colors, ax=ax2, label="")
ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')

plt.suptitle('Albanian Dialect Dataset Analysis', fontsize=18, y=1.05)
plt.tight_layout()

plt.savefig('albanian_dialect_analysis.png', dpi=300)
print("Graphs saved as 'albanian_dialect_analysis.png'")
plt.show()
