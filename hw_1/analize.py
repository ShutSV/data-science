# Применяя различные методы машинного обучения, оценить их производительность и провести анализ результатов
# Загрузить набор данных о населении городов и создайте модель регрессии для прогнозирования роста населения в будущем

# Загрузка и визуальная оценка данных
# Используются стат.данные от Eurostat о численности населения по группам возрастов для 878 городов ЕС
# Описание https://ec.europa.eu/eurostat/cache/metadata/en/urb_esms.htm
# Данные https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/urb_cpop1?format=TSV&compressed=true
# Исходные данные требуют обработки, в тч Data Preprocessing, тк содержат информацию по возрастным группам в виде
# отдельных блоков строк, а не столбцов, и кроме городов также включена инф и в целом по странам,
# и не по всем временным периодам.
# Предварительная обработка в Excel(Numbers): сохранен в формат CSV "estat_urb_cpop1.csv"
# В файле 54647 строк, из которых первая строка заголовок, 54646 строк с данными
# В файле 35 столбца. В первом столбце freq,indic_ur,cities\TIME_PERIOD - коды городов, имена переменных и групп населения,
# во 2-35 - годы с 1989 по 2022.
# Индикаторы и переменные охватывают несколько аспектов качества жизни, например, демографию, жилье, здравоохранение,
# экономическую активность, рынок труда, неравенство доходов, образовательную квалификацию, окружающую среду,
# климат, схемы путешествий, туризм и культурную инфраструктуру.
# Расшифровка кодов переменных с DE1001V до DE1079V приведена в файле "variable_indicators.csv"
# Так как для остальных переменных нет расшифровки, то для анализа эти данные отброшены,
# т.е. оставлены только строки с 1 по 32474, включая заголовок в строке 1.
# Отдельно сохранен файл CSV с кодами и названием городов "urb_esms_an_4.csv"
import pandas as pd
from prepare_data import df
import matplotlib.pyplot as plt


# Формирование данных для обучения
# Используется датасет с полной численностью населения городов ЕС по годам
# Сортировка датасета по убыванию численности населения в 2019 году и вывод данных для крупнейших 5 городов
# Дата 2019 год принята тк это последний год, за который есть данные по всем городам
df_sort = df.sort_values(by="2019", ascending=False)
df_plot = df_sort.head(5)
x = df_plot.columns[1:]
label_x = "годы"
label_y = "население"

fig, axs = plt.subplots(5, 1, figsize=(16, 10), constrained_layout=True)
for i in range(5):
    axs[i].set_title(f"{df_plot.iloc[i, 0]}", fontsize=10)
    axs[i].set_ylabel(label_y, fontsize=6)
    axs[i].set_xlabel(label_y, fontsize=6)
    axs[i].grid(which="major", linewidth=1.2)
    axs[i].grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
    axs[i].plot(x, df_plot.iloc[i, 1:], label=f"")
    axs[i].xaxis.set_tick_params(rotation=90)

plt.show()
