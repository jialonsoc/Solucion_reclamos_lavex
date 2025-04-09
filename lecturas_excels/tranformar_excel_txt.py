import pandas as pd
import os

# Ruta completa al archivo Excel
ruta_excel = os.path.join('c:/Users/jialo/Desktop/Chamba/Resolucion_reclamos_lavex/lecturas_excels', 'Operaciones_Fintoc.xlsx')

# Leer el archivo Excel
df = pd.read_excel(ruta_excel, dtype=str)

# Reemplazar valores NaN con string vacío
df = df.fillna('')

# Convertir todas las columnas a string y limpiar caracteres especiales
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Guardar como archivo de texto en la misma carpeta
ruta_txt = os.path.join('c:/Users/jialo/Desktop/Chamba/Resolucion_reclamos_lavex/lecturas_excels', 'Operaciones_Fintoc.txt')
df.to_csv(ruta_txt, sep='\t', index=False, encoding='utf-8')