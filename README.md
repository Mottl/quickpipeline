# quickpipeline
quickpipeline is a python module for quick preprocessing of features for further use in machine learning tasks

## Usage:
```python
import pandas as pd
import numpy as np
from quickpipeline as QuickPipeline

# prepare example pandas DataFrame:
s1 = pd.Series([1,2,3,np.nan,4,5], dtype=np.float16)
s2 = pd.Series(["A","B",np.nan,"A","C","B"])
y = pd.Series(["yes","yes","no","yes","no","no"])
df = pd.DataFrame({"s1": s1, "s2": s2, "y": y})

# preprocess dataframe:
pipeline = QuickPipeline(y_column_name="y", copy=True)
df_prepared = pipeline.fit_transform(df)
print(df_prepared)
```

Output:
```
    s1  s2_A  s2_B  s2_C  y
0  1.0     1     0     0  1
1  2.0     0     1     0  1
2  3.0     0     0     0  0
3  3.0     1     0     0  1
4  4.0     0     0     1  0
5  5.0     0     1     0  0
```