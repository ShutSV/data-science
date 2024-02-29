from pycaret.datasets import get_data
from pycaret.clustering import *
import mlflow


# Загрузка данных
data = get_data('tweets')
data.info()
# Запуск MLFlow
mlflow.start_run()
# Инициализация среды
s = setup(data, normalize=True, log_experiment=True, experiment_name="xperiment", log_plots=True, session_id=123)

# Доступные модели
print("Доступные модели", "*" * 50)
print(models())
print("*" * 100)
# Создание модели
kmeans = create_model('kmeans', num_clusters=2)
# другие рабочие модели
# dbscan = create_model('dbscan')
# sc = create_model('sc')

# Оценка модели
print(kmeans)
plot_model(kmeans, plot='elbow')
plot_model(kmeans, plot='silhouette')
plot_model(kmeans, plot='tsne')
plot_model(kmeans, plot='distribution')

# Сохранение модели
save_model(kmeans, 'kmeans_model')
# Завершение MLFlow
mlflow.end_run()

# Получение результатов
result = assign_model(kmeans)
# Сохранение результатов
result.to_csv("result.csv")



# ID                                 Name                                          Reference
#
# kmeans                   K-Means Clustering                     sklearn.cluster._kmeans.KMeans
# ap                     Affinity Propagation  sklearn.cluster._affinity_propagation.Affinity...
# meanshift             Mean Shift Clustering              sklearn.cluster._mean_shift.MeanShift
# sc                      Spectral Clustering       sklearn.cluster._spectral.SpectralClustering
# hclust             Agglomerative Clustering  sklearn.cluster._agglomerative.AgglomerativeCl...
# dbscan     Density-Based Spatial Clustering                     sklearn.cluster._dbscan.DBSCAN
# optics                    OPTICS Clustering                     sklearn.cluster._optics.OPTICS
# birch                      Birch Clustering                       sklearn.cluster._birch.Birch
