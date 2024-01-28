import csv
import re
import pandas as pd
from codename_cities import code_names


file_path = "estat_urb_cpop1.csv"
try:
    with open(file_path, "r") as file:
        content = csv.reader(file, skipinitialspace=True, delimiter=";")
        list_content = list(content)
        list_content[0][0] = "Year"
except Exception as e:
    print("Ощибка чтения файла", e)

# ES005C Zaragoza, ES006C Málaga, ES007C Murcia, ES008C Las Palmas
# выборка строк для четырех городов, для которых есть данные по всем годам
# и траспонирование полученного списка для кадого города отдельно

def df_cites(arg):
    df = []
    df.append(list_content[0])
    for i in list_content:
        if i[0][10:] == arg:
            for j in range(1, len(i)):
                text = i[j]
                i[j] = re.sub(r'[^\d]', '', text)
                if i[j] == "":
                    i[j] = 0
                else:
                    i[j] = int(i[j])
            df.append(i)
    df = pd.DataFrame(df[1:], columns=df[0])
    df = df.transpose()
    return df


df_zaragoza = df_cites("ES005C")
# print(df_zaragoza.head(5))
# print("*" * 50)

df_malaga = df_cites("ES006C")
# print(df_malaga.head(5))
# print("*" * 50)

df_murcia = df_cites("ES007C")
# print(df_murcia.head(5))
# print("*" * 50)

df_las_palmas = df_cites("ES008C")
# print(df_las_palmas.head(5))
# print("*" * 50)
