import pandas as pd
import os

# Ruta del archivo Excel
archivo_excel = "C:\\Users\\revueltaap\\Desktop\\Funciones España\\Funciones Inventory.xlsx"
output_folder = "C:\\Users\\revueltaap\\Desktop\\salida_txt"

# Cargar todas las hojas del Excel como un diccionario de DataFrames
hojas = pd.read_excel(archivo_excel, sheet_name=None,skiprows=1)

# Guardar cada hoja en un archivo TXT
for nombre_hoja, df in hojas.items():
    # Crear un nombre de archivo basado en el nombre de la hoja
    nombre_txt = f"{nombre_hoja}.txt"

    # Guardar el contenido en formato texto (separado por tabulaciones)
    output_path = os.path.join(output_folder, nombre_txt)
    df.to_csv(output_path, sep='\t', index=False)

    print(f"✅ Guardado: {nombre_txt}")

#%%

import numpy as np
import os
import json

# Carpeta donde están los txt
carpeta_txt = "C:\\Users\\revueltaap\\Desktop\\salida_txt"

ruta_json ="C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto ihrat\\ihrat\\ihrat\\src\\level_3_analysis\\damage_functions\\damage_functions_dictionary.json"

# Si el JSON ya existe, lo cargamos; si no, lo inicializamos
if os.path.exists(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        funciones_json = json.load(f)
else:
    funciones_json = {"functions": []}

for archivo in os.listdir(carpeta_txt):
    if archivo.endswith(".txt"):
        # Nombre de la función (sin extensión)
        nombre = os.path.splitext(archivo)[0]

        # Cargar datos: suponemos dos columnas x y y
        ruta = os.path.join(carpeta_txt, archivo)
        datos = np.loadtxt(ruta,skiprows=1)  # usa delimiter="," si tus txt son CSV

        x = datos[:, 0]
        values = datos[:, 1]

        # Añadir función solo si no existe ya
        if not any(f["name"] == nombre for f in funciones_json["functions"]):
            # Guardar en JSON con metadatos
            funciones_json["functions"].append({
                "name": nombre,
                "origen": "Huizinga, J., De Moel, H. and Szewczyk, W., Global flood depth-damage functions: Methodology and the database with guidelines, EUR 28552 EN, Publications Office of the European Union, Luxembourg, 2017, ISBN 978-92-79-67781-6, doi:10.2760/16510, JRC105688.",
                "region": "Europe",
                "property type": "Residential building",
                "application": "Relative",
                "type": "interpolation",
                "variables":['x'],
                "x": x.tolist(),
                "values": values.tolist(),
                "interpolation_type": "linear",
                "asset": "Building Content",
                "units": "% damage"
            })


# Guardar el JSON actualizado
with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(funciones_json, f, ensure_ascii=False, indent=4)
