"""
Módulo de evaluación de propiedades drug-like (Regla de Lipinski / Ro5)
y selección de diversidad estructural mediante esqueletos de Bemis-Murcko.
"""

import os
from typing import Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def calcular_violaciones_lipinski(smiles: str) -> int:
    """
    Evalúa los 4 parámetros clásicos de la regla de cinco de Lipinski (Ro5):
    - Peso Molecular <= 500 Da
    - LogP octanol/agua <= 5.0
    - Donadores de enlaces de hidrógeno (HBD) <= 5
    - Aceptores de enlaces de hidrógeno (HBA) <= 10

    Devuelve el número total de violaciones acumuladas.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 99  # Penalización alta para moléculas inválidas

    violaciones = 0
    if Descriptors.MolWt(mol) > 500.0:
        violaciones += 1
    if Descriptors.MolLogP(mol) > 5.0:
        violaciones += 1
    if Descriptors.NumHDonors(mol) > 5:
        violaciones += 1
    if Descriptors.NumHAcceptors(mol) > 10:
        violaciones += 1

    return violaciones


def obtener_scaffold_murcko(smiles: str) -> Optional[str]:
    """
    Extrae el andamiaje canónico central de Bemis-Murcko en formato SMILES.
    Elimina cadenas laterales conservando anillos y enlazadores inter-anillo.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
    except Exception:
        return None


def main():
    # Rutas relativas del repositorio
    input_csv = "../results/candidatos_consenso_top.csv"
    output_dir = "../results"
    
    # Fallback si el archivo tiene el nombre histórico del notebook
    if not os.path.exists(input_csv):
        input_csv = "../results/TOP_inhibidores_FabI_CMNPD.csv"

    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Cargando candidatos con consenso pChEMBL >= 6.0 desde: {input_csv}")
    df_candidatos = pd.read_csv(input_csv)
    print(f"       -> Compuestos de partida: {len(df_candidatos)}")

    # 1. Análisis de propiedades de Lipinski
    print("[2/4] Calculando violaciones de Lipinski (Ro5)...")
    df_candidatos["Violaciones_Lipinski"] = df_candidatos["Smiles"].apply(
        calcular_violaciones_lipinski
    )

    # Registro didáctico: filtro estricto vs flexible para la memoria
    estrictos = (df_candidatos["Violaciones_Lipinski"] == 0).sum()
    flexibles = (df_candidatos["Violaciones_Lipinski"] <= 1).sum()
    print(f"       -> Con 0 violaciones (estricto): {estrictos}")
    print(f"       -> Con <= 1 violación (adaptado a productos naturales): {flexibles}")

    # Filtrado permitiendo un máximo de 1 infracción
    df_lipinski = df_candidatos[df_candidatos["Violaciones_Lipinski"] <= 1].copy()
    ruta_lipinski = os.path.join(output_dir, "CMNPD_Lipinski_Flexible.csv")
    df_lipinski.to_csv(ruta_lipinski, index=False)
    print(f"       -> Guardado subconjunto Lipinski en: {ruta_lipinski}")

    # 2. Descomposición y selección por andamios de Murcko
    print("[3/4] Extrayendo esqueletos de Bemis-Murcko para análisis de quimiotipos...")
    df_lipinski["Scaffold"] = df_lipinski["Smiles"].apply(obtener_scaffold_murcko)
    df_lipinski = df_lipinski.dropna(subset=["Scaffold"]).copy()

    # Ordenar por el score consenso de mayor a menor
    df_lipinski = df_lipinski.sort_values(by="Media_Predicciones", ascending=False)

    # Conservar únicamente la mejor molécula de cada familia molecular
    df_representantes = df_lipinski.drop_duplicates(
        subset=["Scaffold"], keep="first"
    ).reset_index(drop=True)

    # 3. Exportación del conjunto élite para acoplamiento molecular
    ruta_elite = os.path.join(output_dir, "CMNPD_Representantes_Elite.csv")
    df_representantes.to_csv(ruta_elite, index=False)

    print("\n" + "=" * 65)
    print("RESUMEN DE DIVERSIFICACIÓN ESTRUCTURAL Y LIKENESS")
    print("=" * 65)
    print(f"Candidatos iniciales (pChEMBL >= 6.0):      {len(df_candidatos)}")
    print(f"Candidatos aprobados Lipinski (<= 1 viol):  {len(df_lipinski)}")
    print(f"Familias moleculares (Scaffolds) únicas:    {len(df_representantes)}")
    print(f"Archivo de representantes élite:             {ruta_elite}")
    print("=" * 65)

    print("\nTop 5 representantes priorizados:")
    columnas_resumen = ["Smiles", "Scaffold", "Media_Predicciones"]
    print(df_representantes[columnas_resumen].head(5).to_string(index=False))
    print("\n[4/4] ¡Filtrado de biodisponibilidad y selección de scaffolds completados con éxito!")


if __name__ == "__main__":
    main()