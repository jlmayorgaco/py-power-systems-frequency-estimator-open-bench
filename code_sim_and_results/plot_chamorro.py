import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo CSV
# Si el CSV no tiene encabezados, usa: names=['Tiempo', 'Valor_B1']
df = pd.read_csv('Dataset_Chamorro.csv')

# 2. Extraer los datos 
# (Asumiendo que las columnas se llaman 'Tiempo' y 'Valor_B1' o son las únicas dos)
t = df.iloc[:, 0]  # Primera columna
b1 = df.iloc[:, 1] # Segunda columna

# 3. Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(t, b1, label='b1 vs t', color='blue', linewidth=1.5)

# Personalización
plt.title('Gráfica de t vs b1 (Desde MATLAB a Python)')
plt.xlabel('Tiempo (t)')
plt.ylabel('Valor (b1)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Mostrar la gráfica
plt.show()