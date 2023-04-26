from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

''' KNN '''

# Carga el conjunto de datos
data = pd.read_csv("cleaned_creditcard.csv")

# Divide el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Class", axis=1), data["Class"], test_size=0.3, random_state=42)

# Entrena el modelo k-NN con k=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del clasificador k-NN con k=%d: %.2f%%" % (k, accuracy*100))


''' K-Means '''

# Carga el conjunto de datos
data = pd.read_csv("cleaned_creditcard.csv")

# Selecciona las características relevantes
X = data[["Amount", "Time"]]

# Crea un modelo K-means con k=2
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)

# Entrena el modelo
kmeans.fit(X)

# Asigna una etiqueta de grupo a cada observación
labels = kmeans.predict(X)

# Visualiza los grupos en un diagrama de dispersión
plt.scatter(X["Time"], X["Amount"], c=labels)
plt.xlabel("Tiempo")
plt.ylabel("Monto")
plt.show()
