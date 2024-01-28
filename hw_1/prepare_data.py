import csv
import re
import pandas as pd
from codename_cities import code_names


# выборка строк, содержащих код DE1001V, т.е. полное население городов ЕС, и относящихся к городам из списка code_names
file_path = "estat_urb_cpop1.csv"
list_for_df = []
try:
    with open(file_path, "r") as file:
        content = csv.reader(file, skipinitialspace=True, delimiter=";")
        list_content = list(content)
except Exception as e:
    print("Ощибка чтения файла", e)


list_content[0][0] = "Cities"
list_for_df.append(list_content[0])
for i in list_content:
    if i[0][2:9] == "DE1001V" and i[0][10:] in code_names.keys():
        i[0] = code_names.get(i[0][10:])
        for j in range(1, len(i)):
            text = i[j]
            i[j] = re.sub(r'[^\d]', '', text)
            if i[j] == "":
                i[j] = 0
            else:
                i[j] = int(i[j])
        list_for_df.append(i)

df = pd.DataFrame(list_for_df[1:], columns=list_for_df[0])
# print(df.describe())
# print(df.info())
# print(df.head(5))
