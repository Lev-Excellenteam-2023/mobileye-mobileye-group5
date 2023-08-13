import os
import json
from pathlib import Path
import pandas as pd

original_csv_path = os.path.join('..', 'data', 'tfls.csv')
filtered_csv_path = os.path.join('..', 'data', 'filtered_tfls.csv')
df = pd.read_csv(original_csv_path)
filtered_df = pd.DataFrame(columns=df.columns)
print(pd. __version__)

tl_label = 'traffic light'
filtered_idx = 0
for index, row in df.iterrows():
    gt_data = json.loads(Path(os.path.join('..', 'data',row['json_path'])).read_text())
    objects = [o for o in gt_data['objects'] if o['label'] == tl_label]
    if len(objects):
        # Add the row to the filtered DataFrame if it satisfies the condition
        filtered_df.loc[filtered_idx] = row
        filtered_idx += 1

# Save the new CSV with filtered rows
filtered_df.to_csv(filtered_csv_path, index=False)