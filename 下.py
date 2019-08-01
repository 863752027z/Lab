import numpy as np
import pandas as pd

def save_data_to_excel(data):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('E:/Save_Excel.xlsx')
    data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
    writer.save()


a = np.empty([1, 3], dtype=np.float32)
print(a)
b = np.array([[1,2,3]])
print(b)
c = np.concatenate([a, b])
print(c)