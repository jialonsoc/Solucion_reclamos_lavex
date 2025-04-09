import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, expon, gamma, lognorm, weibull_min
from scipy.optimize import curve_fit
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

# Leer datos
df = pd.read_csv('lecturas_excels/Operaciones_Fintoc.txt', sep='\t', skiprows=3)
df['tiempo_segundos'] = df['tiempo [hh:mm:ss]'].apply(tiempo_a_segundos)
df_succeeded = df[df['status'] == 'succeeded'].copy()
muestra = df_succeeded['tiempo_segundos'].dropna().values

# Función para ajustar y evaluar distribuciones
def ajustar_distribucion(datos, distribucion, nombre):
    # Ajustar parámetros
    params = distribucion.fit(datos)
    
    # Calcular estadístico KS y valor p
    ks_stat, p_value = stats.kstest(datos, distribucion.cdf, args=params)
    
    # Calcular AIC y BIC
    log_likelihood = np.sum(distribucion.logpdf(datos, *params))
    n_params = len(params)
    n_samples = len(datos)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    
    return {
        'nombre': nombre,
        'params': params,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood
    }

# Lista de distribuciones a probar
distribuciones = [
    (stats.expon, 'Exponencial'),
    (stats.gamma, 'Gamma'),
    (stats.lognorm, 'Lognormal'),
    (stats.weibull_min, 'Weibull'),
    (stats.norm, 'Normal')
]

# Ajustar todas las distribuciones
resultados = []
for dist, nombre in distribuciones:
    try:
        resultado = ajustar_distribucion(muestra, dist, nombre)
        resultados.append(resultado)
    except Exception as e:
        print(f"Error ajustando {nombre}: {str(e)}")

# Crear DataFrame con resultados
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('aic')

# Imprimir resultados en formato de tabla
print("\nComparación de Ajustes de Distribuciones:")
print("=======================================")
print("\n{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
    "Distribución", "AIC", "BIC", "KS Stat", "Valor p", "Ajuste"))
print("-" * 90)

for _, row in df_resultados.iterrows():
    # Determinar calidad del ajuste
    if row['p_value'] > 0.05:
        ajuste = "Bueno"
    elif row['p_value'] > 0.01:
        ajuste = "Regular"
    else:
        ajuste = "Malo"
    
    print("{:<15} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.4f} {:<15}".format(
        row['nombre'],
        row['aic'],
        row['bic'],
        row['ks_stat'],
        row['p_value'],
        ajuste
    ))

# Imprimir recomendación detallada
print("\nRecomendación:")
print("=============")
mejor_dist = df_resultados.iloc[0]
print(f"\nLa mejor distribución es: {mejor_dist['nombre']}")
print(f"\nRazones:")
print(f"1. Tiene el menor AIC ({mejor_dist['aic']:.2f})")
print(f"2. Tiene el menor BIC ({mejor_dist['bic']:.2f})")
print(f"3. Estadístico KS: {mejor_dist['ks_stat']:.4f}")
print(f"4. Valor p: {mejor_dist['p_value']:.4f}")

if mejor_dist['p_value'] > 0.05:
    print("\nInterpretación:")
    print("La distribución se ajusta bien a los datos (valor p > 0.05)")
    print("Podemos usar esta distribución para modelar los tiempos de espera")
else:
    print("\nInterpretación:")
    print("Aunque es la mejor de las opciones probadas, el valor p sugiere que")
    print("ninguna de las distribuciones probadas se ajusta perfectamente a los datos")
    print("Esto podría indicar que los tiempos de espera siguen un patrón más complejo")

# Crear gráfico comparativo
plt.figure(figsize=(15, 10))

# Histograma de datos
plt.subplot(2, 1, 1)
sns.histplot(muestra, bins=30, stat='density', alpha=0.6, label='Datos')

# Graficar las distribuciones ajustadas
x = np.linspace(min(muestra), max(muestra), 1000)
for _, row in df_resultados.iterrows():
    dist = next(d for d, n in distribuciones if n == row['nombre'])
    y = dist.pdf(x, *row['params'])
    plt.plot(x, y, label=f"{row['nombre']} (AIC: {row['aic']:.2f})")

plt.title('Comparación de Ajustes de Distribuciones', fontsize=14)
plt.xlabel('Tiempo (segundos)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()

# Gráfico Q-Q para la mejor distribución
plt.subplot(2, 1, 2)
mejor_dist = next(d for d, n in distribuciones if n == df_resultados.iloc[0]['nombre'])
params = df_resultados.iloc[0]['params']
stats.probplot(muestra, dist=mejor_dist, sparams=params, plot=plt)
plt.title(f'Gráfico Q-Q para {df_resultados.iloc[0]["nombre"]}', fontsize=14)

plt.tight_layout()
plt.show() 