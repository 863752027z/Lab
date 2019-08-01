import numpy as np
import pandas as pd
import datetime

print(datetime.datetime.now())
path = 'E:/Save_Excel.xlsx'
sheet = 'page_1'
df = pd.read_excel(path, sheet)
M = np.array(df)
M = np.delete(M,0,)
print(M)
print(M.shape)
print(datetime.datetime.now())

