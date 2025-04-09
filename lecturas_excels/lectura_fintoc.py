import numpy as np
from scipy.stats import chi2, expon, kstest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

# Definir función para convertir tiempo a segundos
def tiempo_a_segundos(tiempo_str):
    # Manejar valores nulos o vacíos
    if pd.isna(tiempo_str) or tiempo_str == '':
        return None
    
    # Convertir tiempo en formato hh:mm:ss a segundos
    try:
        h, m, s = map(int, tiempo_str.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None

# Leer el archivo txt
df = pd.read_csv('lecturas_excels/Operaciones_Fintoc.txt', sep='\t', skiprows=3)

# Convertir tiempos a segundos para todos los registros
df['tiempo_segundos'] = df['tiempo [hh:mm:ss]'].apply(tiempo_a_segundos)

# Separar datos por estado
df_succeeded = df[df['status'] == 'succeeded'].copy()
df_failed = df[df['status'] == 'failed'].copy()
df_waiting = df[df['status'] == 'waiting'].copy()

# Obtener la muestra de tiempos exitosos
muestra_succeeded = [t for t in df_succeeded['tiempo_segundos'].tolist() if t is not None]

def estimar_lambda_intervalo(muestra, alpha=0.05):
    n = len(muestra)
    x_barra = np.mean(muestra)
    lambda_hat = 1 / x_barra

    gl = 2 * n  # grados de libertad
    chi2_lower = chi2.ppf(alpha / 2, gl)
    chi2_upper = chi2.ppf(1 - alpha / 2, gl)

    # Intervalo de confianza
    intervalo_inf = chi2_lower / (2 * n * x_barra)
    intervalo_sup = chi2_upper / (2 * n * x_barra)

    return lambda_hat, (intervalo_inf, intervalo_sup)

# Calcular estimaciones (solo con transacciones exitosas)
lambda_emv, intervalo = estimar_lambda_intervalo(muestra_succeeded)
tiempo_esperado = 1/lambda_emv
intervalo_tiempo = (1/intervalo[1], 1/intervalo[0])

# Configuración de estilo para gráficos más atractivos
sns.set_style("whitegrid")
plt.figure(figsize=(16, 10))

# 1. Subplot superior: Histograma con KDE y curva exponencial
plt.subplot(2, 1, 1)

# Histograma para transacciones exitosas
sns.histplot(df_succeeded['tiempo_segundos'].dropna(), bins=20, kde=True, 
             color='steelblue', alpha=0.6, label='Exitosas', stat='density')

# Histograma para transacciones fallidas
if not df_failed['tiempo_segundos'].isna().all():
    sns.histplot(df_failed['tiempo_segundos'].dropna(), bins=20, kde=True, 
                color='tomato', alpha=0.5, label='Fallidas', stat='density')

# Curva de la distribución exponencial teórica
x = np.linspace(0, max(df['tiempo_segundos'].dropna()) * 1.1, 1000)
y = expon.pdf(x, scale=1/lambda_emv)
plt.plot(x, y, 'g-', linewidth=2, label=f'Exp(λ={lambda_emv:.4f})')

# Agregar línea vertical para el tiempo esperado
plt.axvline(x=tiempo_esperado, color='darkgreen', linestyle='--', 
           linewidth=2, label=f'Tiempo esperado: {tiempo_esperado:.2f}s')

# Sombrear el intervalo de confianza
plt.axvspan(intervalo_tiempo[0], intervalo_tiempo[1], alpha=0.2, color='green',
           label=f'IC 95%: ({intervalo_tiempo[0]:.2f}, {intervalo_tiempo[1]:.2f})s')

plt.title('Distribución de Tiempos de Espera con Ajuste Exponencial', fontsize=14)
plt.xlabel('Tiempo (segundos)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend(fontsize=10)
plt.xlim(0, np.percentile(df['tiempo_segundos'].dropna(), 95) * 1.2)

# 2. Subplot inferior: Boxplot comparativo
plt.subplot(2, 1, 2)

# Crear boxplot comparativo solo para succeeded y failed
df_filtered = df[df['status'].isin(['succeeded', 'failed'])]
sns.boxplot(x='status', y='tiempo_segundos', data=df_filtered, 
           hue='status', palette={'succeeded': 'steelblue', 'failed': 'tomato'},
           legend=False)

# Superponer swarmplot para ver distribución de puntos
sns.swarmplot(x='status', y='tiempo_segundos', data=df_filtered, 
             hue='status', palette={'succeeded': 'darkblue', 'failed': 'darkred'},
             alpha=0.7, size=4, legend=False)

# Personalizar boxplot
plt.title('Comparación de Tiempos por Estado de Transacción', fontsize=14)
plt.xlabel('Estado', fontsize=12)
plt.ylabel('Tiempo (segundos)', fontsize=12)
plt.xticks([0, 1], ['Exitosas', 'Fallidas'])
plt.ylim(0, np.percentile(df_filtered['tiempo_segundos'].dropna(), 95) * 1.2)

# Agregar estadísticas en el gráfico
plt.text(0, np.percentile(df_succeeded['tiempo_segundos'].dropna(), 95), 
         f"n={len(df_succeeded)}\nMedia={df_succeeded['tiempo_segundos'].mean():.2f}s", 
         ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
