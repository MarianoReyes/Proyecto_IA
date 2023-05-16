'''
K-means para la deteccion de usuarios posiblemente fraudulentos
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar datos en un DataFrame
df = pd.read_csv('card_transdata.csv')

# Seleccionar características relevantes (opcional)
selected_features = df[['distance_from_home',
                        'distance_from_last_transaction', 'ratio_to_median_purchase_price']]

# Preprocesamiento de datos: escalar características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

# Elegir el número de clusters (opcional)
num_clusters = 3

# Aplicar algoritmo de clustering K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Obtener las etiquetas de cluster para cada instancia
cluster_labels = kmeans.labels_

# Añadir las etiquetas de cluster al DataFrame original
df['cluster'] = cluster_labels

# Calcular la proporción de instancias fraudulentas en cada cluster
fraudulent_ratio = df.groupby('cluster')['fraud'].mean()

# Identificar el cluster con mayor proporción de instancias fraudulentas
suspicious_cluster = fraudulent_ratio.idxmax()

# Obtener las instancias en el cluster sospechoso
suspicious_instances = df[df['cluster'] == suspicious_cluster]

# Crear el perfil de posibles personas fraudulentas
fraudulent_profile = suspicious_instances.drop(['cluster'], axis=1)

# Guardar los perfiles fraudulentos en un nuevo archivo CSV
fraudulent_profile.to_csv('perfiles_fraudulentos.csv', index=False)
