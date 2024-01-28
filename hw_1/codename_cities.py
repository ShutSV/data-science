import os
import pandas as pd

code_names = dict()
# попытка получить данные из файла CSV, с загрузкой в DataFrame
file_path = "urb_esms_an_4.csv"
if not os.path.exists(file_path):
    print("Такой файл не существует")
try:
    df_read = pd.read_csv(file_path,
                     header=0,
                     index_col=False,
                     sep=';')
    for index, row in df_read.iterrows():
        if row["CODE"][-1] == "C":
            # print(row["CODE"], row["NAME"])
            code_names[row["CODE"]] = row["NAME"]
except Exception as e:
    print("Ошибка чтения из файла", e)



