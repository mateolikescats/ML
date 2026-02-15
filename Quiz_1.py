import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names

print('Descripción del dataset:')
print(california.DESCR[:1500])

n_observaciones, n_caracteristicas = X.shape
primera_observacion = X[0]

print(f"a) Numero de registros (n): {n_observaciones}")

print(f"b) Numero de caracteristicas por observacion: {n_caracteristicas}")

print("c) Vector de caracteristicas (x) para la primera observacion:")
print(primera_observacion)
print("\n   Vector x en forma de columna (notacion matematica):")
for i, nombre in enumerate(feature_names):
    print(f"   [{primera_observacion[i]:>8.4f}]  <- {nombre}")

for i, name in enumerate(california.feature_names, 1):
    print(f"{i}. {name}")
desc_text = california.DESCR
start_idx = desc_text.find("Attribute Information")
end_idx = desc_text.find("Missing Attribute Values")

if start_idx != -1:
    print(desc_text[start_idx:end_idx].strip())
else:
    print("No se encontró la sección detallada en DESCR.")