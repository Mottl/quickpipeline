# quickpipeline
quickpipeline is a python module for quick preprocessing of features for further use in machine learning tasks

## Usage:

  import pandas as pd
  import numpy as np
  from quickpipeline as QuickPipeline

  # prepare example pandas DataFrame:
  s1 = pd.Series([1,2,3,np.nan,4,5], dtype=np.float16)
  s2 = pd.Series(["A","B",np.nan,"A","C","B"])
  df = pd.DataFrame({"s1": s1, "s2": s2})

  # preprocess dataframe:
  pipeline = QuickPipeline(copy=True)
  df_prepared = pipeline.fit_transform(df)
  print(df_prepared)