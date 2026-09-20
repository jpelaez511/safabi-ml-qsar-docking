"""
Módulo de preparación conformacional 3D y minimización energética de los
candidatos Top élite de CMNPD para acoplamiento molecular (Docking).
"""

import os
from typing import Dict, Any
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def generar_conformacion_3d(smiles: str, max_iters: int = 200) -> Chem.Mol:
    """
    Convierte un SMILES 2D en una estructura conformacional 3D minimizada:
    1. Añade hidrógenos explícitos a la topología molecular.
    2. Embebe coordenadas tridimensionales mediante el algoritmo ETKDG.
    3. Minimiza la energía del confórmero usando el campo de fuerzas MMFF94.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES inválido no procesable por RDKit: {smiles}")

    # Protonación y adición de hidrógenos explícitos
    mol_h = Chem.AddHs(mol)

    # Generación conformacional inicial con ETKDG
    parametros_etkdg = AllChem.ETKDGv3()
    parametros_etkdg.randomSeed = 42
    status_embed = AllChem.EmbedMolecule(mol_h, parametros_etkdg)

    # Fallback clásico si falla la versión 3 en sistemas altamente flexibles
    if status_embed != 0:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

    # Minimización energética MMFF94
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=max_iters)
    return mol_h


def exportar_ligandos_top(df_elite: pd.DataFrame, top_n: int = 5, output_dir: str = "../results") -> pd.DataFrame:
    """
    Selecciona los mejores N representantes de diferentes scaffolds,
    genera sus confórmeros 3D minimizados y los exporta a archivos SDF individuales.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_top = df_elite.sort_values(by="Media_Predicciones", ascending=False).head(top_n).reset_index(drop=True)

    print(f"\n[1/2] Generando y minimizando confórmeros 3D para el Top {top_n} élite...")

    for rank, row in df_top.iterrows():
        posicion = rank + 1
        smiles = row["Smiles"]
        score = row["Media_Predicciones"]

        # Modelado 3D
        mol_3d = generar_conformacion_3d(smiles, max_iters=200)

        # Nombre estandarizado de archivo
        nombre_sdf = f"Top_{posicion}_Pred_{score:.2f}.sdf"
        ruta_archivo = os.path.join(output_dir, nombre_sdf)

        writer = Chem.SDWriter(ruta_archivo)
        writer.write(mol_3d)
        writer.close()

        print(f"       -> [Top {posicion}] Guardado: {nombre_sdf} (pChEMBL pred: {score:.2f})")

    return df_top


def reportar_candidato_lider(df_top: pd.DataFrame, posicion_lider: int = 3):
    """
    Muestra la ficha técnica detallada del compuesto cabeza de serie (Top 3 - Enisorina E)
    para su trazabilidad y uso en simulaciones de docking en UCSF Chimera.
    """
    idx = posicion_lider - 1
    if idx >= len(df_top):
        print(f"Advertencia: No existe la posición {posicion_lider} en el dataframe.")
        return

    candidato = df_top.iloc[idx]

    print("\n" + "=" * 65)
    print(f"FICHA TÉCNICA: COMPUESTO CABEZA DE SERIE (TOP {posicion_lider} - ENISORINA E)")
    print("=" * 65)
    for columna, valor in candidato.items():
        if isinstance(valor, float):
            print(f"{columna:<25}: {valor:.4f}")
        else:
            print(f"{columna:<25}: {valor}")
    print("=" * 65)
    print(f"\nSMILES canónico de consulta:\n{candidato['Smiles']}")


def main():
    # Rutas relativas estándar
    input_csv = "../results/CMNPD_Representantes_Elite.csv"
    output_dir = "../results"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada '{input_csv}'. "
            "Asegúrate de ejecutar previamente el script 05."
        )

    print(f"Cargando representantes moleculares desde: {input_csv}")
    df_elite = pd.read_csv(input_csv)

    # 1. Preparar confórmeros 3D y guardar en SDF
    df_top_5 = exportar_ligandos_top(df_elite, top_n=5, output_dir=output_dir)

    # 2. Desglose del candidato principal (Top 3)
    reportar_candidato_lider(df_top_5, posicion_lider=3)

    print(f"\n[2/2] ¡Preparación 3D completada! Archivos listos para acoplamiento molecular en UCSF Chimera.")


if __name__ == "__main__":
    main()