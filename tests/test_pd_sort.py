import pandas as pd
df = pd.DataFrame({"text": ["A", "B"], "other": [1, 2]})
combined = pd.concat([df, df.copy()]).sort_index(kind='stable')
print(combined)
