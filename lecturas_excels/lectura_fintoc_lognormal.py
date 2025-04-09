import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Función para convertir tiempo a segundos
def tiempo_a_segundos(tiempo_str):
    if pd.isna(tiempo_str) or tiempo_str == '':
        return None
    try:
        h, m, s = map(int, tiempo_str.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None

# Función para calcular intervalo de confianza de la media en distribución lognormal
def intervalo_media_lognormal(muestra, alpha=0.05):
    log_muestra = np.log(muestra)
    n = len(log_muestra)

    mu_hat = np.mean(log_muestra)
    sigma_hat = np.std(log_muestra, ddof=0)  # EMV

    # Estimación de la media de X
    media_hat = np.exp(mu_hat + 0.5 * sigma_hat**2)

    # Margen de error con delta method
    z = norm.ppf(1 - alpha / 2)
    se_log_mean = np.sqrt((sigma_hat**2) / n + (sigma_hat**4) / (2 * (n - 1)))
    margen = z * se_log_mean

    # Intervalo de confianza para E[X]
    inferior = np.exp(mu_hat + 0.5 * sigma_hat**2 - margen)
    superior = np.exp(mu_hat + 0.5 * sigma_hat**2 + margen)

    return media_hat, (inferior, superior)

# Leer datos
df = pd.read_csv('lecturas_excels/Operaciones_Fintoc.txt', sep='\t', skiprows=3)
df['tiempo_segundos'] = df['tiempo [hh:mm:ss]'].apply(tiempo_a_segundos)

# Separar datos por estado
df_succeeded = df[df['status'] == 'succeeded'].copy()
df_failed = df[df['status'] == 'failed'].copy()
df_waiting = df[df['status'] == 'waiting'].copy()

# Obtener la muestra de tiempos exitosos
muestra_succeeded = df_succeeded['tiempo_segundos'].dropna().values

# Calcular estadísticas descriptivas
media, intervalo_media = intervalo_media_lognormal(muestra_succeeded)
mu_hat = np.mean(np.log(muestra_succeeded))
sigma_hat = np.std(np.log(muestra_succeeded), ddof=0)

print("\nAnálisis de Tiempos de Espera (Distribución Lognormal)")
print("===================================================")
print(f"\nEstadísticas Descriptivas:")
print(f"Media estimada: {media:.2f} segundos")
print(f"Intervalo de confianza al 95%: ({intervalo_media[0]:.2f}, {intervalo_media[1]:.2f}) segundos")
print(f"Parámetro mu (log-normal): {mu_hat:.4f}")
print(f"Parámetro sigma (log-normal): {sigma_hat:.4f}")

# Crear gráficos
plt.figure(figsize=(15, 10))

# 1. Histograma con ajuste lognormal
plt.subplot(2, 1, 1)

# Graficar histograma para transacciones exitosas
sns.histplot(muestra_succeeded, bins=30, stat='density', alpha=0.6, 
            color='steelblue', label='Exitosas')

# Graficar histograma para transacciones fallidas
muestra_failed = df_failed['tiempo_segundos'].dropna().values
sns.histplot(muestra_failed, bins=30, stat='density', alpha=0.6, 
            color='tomato', label='Fallidas')

# Graficar PDF lognormal para transacciones exitosas
x = np.linspace(min(muestra_succeeded), max(muestra_succeeded), 1000)
pdf_lognorm = stats.lognorm.pdf(x, s=sigma_hat, scale=np.exp(mu_hat))
plt.plot(x, pdf_lognorm, 'darkblue', label='Ajuste Lognormal (Exitosas)')

# Agregar línea vertical para la media
plt.axvline(x=media, color='darkgreen', linestyle='--', 
           linewidth=2, label=f'Media: {media:.2f}s')

# Sombrear el intervalo de confianza
plt.axvspan(intervalo_media[0], intervalo_media[1], alpha=0.2, color='green',
           label=f'IC 95%: ({intervalo_media[0]:.2f}, {intervalo_media[1]:.2f})s')

plt.title('Distribución de Tiempos de Espera', fontsize=14)
plt.xlabel('Tiempo (segundos)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()

# 2. Boxplot comparativo
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
plt.text(1, np.percentile(df_failed['tiempo_segundos'].dropna(), 95), 
         f"n={len(df_failed)}\nMedia={df_failed['tiempo_segundos'].mean():.2f}s", 
         ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Prueba de bondad de ajuste
ks_stat, p_value = stats.kstest(muestra_succeeded, 'lognorm', args=(sigma_hat,), alternative='two-sided')
print(f"\nPrueba de bondad de ajuste (Kolmogorov-Smirnov):")
print(f"Estadístico KS: {ks_stat:.4f}")
print(f"Valor p: {p_value:.4f}")

if p_value > 0.05:
    print("\nInterpretación:")
    print("La distribución lognormal es un buen ajuste para los datos (valor p > 0.05)")
else:
    print("\nInterpretación:")
    print("La distribución lognormal no se ajusta perfectamente a los datos")
    print("Esto podría indicar que los tiempos de espera siguen un patrón más complejo") 