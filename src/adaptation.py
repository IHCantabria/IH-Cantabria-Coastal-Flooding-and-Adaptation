import copy
from pathlib import Path
import rasterio
from rasterio.mask import mask


def reduce_exp_value(geojson_path, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    result = copy.deepcopy(geojson)
    for feature in result["features"]:
        if "EXP_VALUE" in feature["properties"]:
            feature["properties"]["EXP_VALUE"] = round(feature["properties"]["EXP_VALUE"] * 0.9)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Archivo exportado: {output_path}")

def improve_build_res(geojson_path, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)


    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    for feature in geojson["features"]:
        if "dam_fun" in feature["properties"] and feature["properties"]["dam_fun"] is not None:
            feature["properties"]["dam_fun"] = str(feature["properties"]["dam_fun"]) + "_reduced"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"Archivo exportado: {output_path}")

import json

def add_coastal_protection(geojson_path, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)


    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Encontrar el feature con menor X
    min_feature = min(
        geojson["features"],
        key=lambda f: f["geometry"]["coordinates"][0]
    )

    # Restar 1.0 a los campos con nombres numéricos
    for key, value in min_feature["properties"].items():
        if all(part.isnumeric() for part in key.split("_")) and value is not None:
            min_feature["properties"][key] = max(0, value - 1.0)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"Archivo exportado: {output_path}")



def retreat_buildings(geojson_path, tif_path, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar edificios
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    total_original = len(geojson["features"])
    filtered_features = []

    with rasterio.open(tif_path) as src:

        for feature in geojson["features"]:

            try:
                out_image, _ = mask(
                    src,
                    [feature["geometry"]],
                    crop=True,
                    filled=False,
                    all_touched=True
                )

                # Primera banda
                data = out_image[0]

                # Quita nodata y píxeles enmascarados
                pixels = data.compressed()

                # Si hay al menos un píxel inundado → eliminar
                inundado = len(pixels) > 0

                # Conservar solo edificios NO inundados
                if not inundado:
                    filtered_features.append(feature)

            except Exception as e:
                print(f"Error: {e}")

                # Si hay error (fuera del raster, geometría rara...)
                # mejor conservar el edificio
                filtered_features.append(feature)

    geojson["features"] = filtered_features

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            geojson,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Polígonos originales: {total_original}")
    print(f"Polígonos eliminados: {total_original-len(filtered_features)}")
    print(f"Polígonos conservados: {len(filtered_features)}")
    print(f"Archivo exportado: {output_path}")