"""Módulo de extracción y selección de características moleculares (Descriptores Fisicoquímicos y MACCS Keys) para saFabI."""

import os
from typing import List, Tuple
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold

# Subconjunto de 78 descriptores fisicoquímicos seleccionados (NCATS / JCIM)
LISTA_DESCRIPTORES_NCATS = [
    "MolLogP",
    "MolWt",
    "TPSA",
    "LabuteASA",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumHeteroatoms",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "FractionCSP3",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Chi4v",
    "Chi1n",
    "Chi2n",
    "Chi3n",
    "Chi4n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SMR_VSA1",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SMR_VSA10",
    "PEOE_VSA1",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
]


def _calc_desc_individual(
    smi: str, calculator: MoleculeDescriptors.MolecularDescriptorCalculator
) -> List[float]:
  """Calcula los descriptores fisicoquímicos para un SMILES individual."""
  mol = Chem.MolFromSmiles(smi)
  if mol is None:
    return [np.nan] * len(LISTA_DESCRIPTORES_NCATS)
  return list(calculator.CalcDescriptors(mol))


def calcular_descriptores_fisicoquimicos(
    smiles_list: List[str], n_jobs: int = -1
) -> pd.DataFrame:
  """Calcula en paralelo el subconjunto de descriptores fisicoquímicos NCATS."""
  calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
      LISTA_DESCRIPTORES_NCATS
  )
  descriptors = Parallel(n_jobs=n_jobs, prefer="threads")(
      delayed(_calc_desc_individual)(smi, calculator) for smi in smiles_list
  )
  df_desc = pd.DataFrame(descriptors, columns=LISTA_DESCRIPTORES_NCATS)
  # Eliminar columnas con valores nulos si algún cálculo falla
  return df_desc.dropna(axis=1, how="any")


def calcular_maccs_keys(smiles_list: List[str]) -> pd.DataFrame:
  """Calcula las huellas estructurales binarias MACCS Keys (167 bits) para una lista de SMILES."""

  def _maccs_individual(smi: str) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      return [0] * 167
    return list(MACCSkeys.GenMACCSKeys(mol))

  maccs_list = [_maccs_individual(smi) for smi in smiles_list]
  columnas = [f"MACCS_{i}" for i in range(167)]
  return pd.DataFrame(maccs_list, columns=columnas)


def filtrar_features(
    df: pd.DataFrame,
    nombre_dataset: str,
    target_col: str = "pChEMBL_Value",
    var_threshold: float = 0.1,
    corr_threshold: float = 0.80,
) -> pd.DataFrame:
  """Aplica reducción de dimensionalidad mediante:

  1. Filtro de varianza mínima (VarianceThreshold).
  2. Filtro de colinealidad de Pearson en el triángulo superior.
  """
  print(f"\n--- FILTRANDO DATASET: {nombre_dataset} ---")
  df_features = df.drop(columns=[target_col])
  print(f"Descriptores iniciales: {df_features.shape[1]}")

  # 1. Filtro de Varianza
  selector = VarianceThreshold(threshold=var_threshold)
  selector.fit(df_features)
  cols_var = df_features.columns[selector.get_support()]
  df_var = df_features[cols_var]
  print(
      f"Descriptores tras Varianza (< {var_threshold}): {df_var.shape[1]}"
  )

  # 2. Filtro de Correlación de Pearson
  matriz_corr = df_var.corr(method="pearson").abs()
  upper_tri = matriz_corr.where(
      np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
  )
  to_drop = [
      col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)
  ]
  df_final = df_var.drop(columns=to_drop)
  print(
      f"Descriptores tras Correlación (> {corr_threshold}):"
      f" {df_final.shape[1]}"
  )

  # Reintegrar la variable dependiente
  return pd.concat([df_final, df[[target_col]]], axis=1)


def main():
  input_csv = "../data/processed/chembl_curado.csv"
  output_dir = "../data/processed"
  os.makedirs(output_dir, exist_ok=True)

  print("[1/4] Leyendo dataset curado...")
  df_curado = pd.read_csv(input_csv)
  smiles = df_curado["Smiles"].tolist()
  target = df_curado["pChEMBL Value"]

  print("[2/4] Calculando descriptores fisicoquímicos en paralelo...")
  df_fisicoq = calcular_descriptores_fisicoquimicos(smiles, n_jobs=-1)
  df_fisicoq["pChEMBL_Value"] = target

  print("[3/4] Calculando huellas estructurales MACCS Keys (167 bits)...")
  df_maccs = calcular_maccs_keys(smiles)
  df_maccs["pChEMBL_Value"] = target

  print("[4/4] Aplicando selección de variables (Varianza y Correlación)...")
  df_fisicoq_filtrado = filtrar_features(
      df_fisicoq,
      "FISICOQUÍMICOS",
      target_col="pChEMBL_Value",
      var_threshold=0.1,
      corr_threshold=0.80,
  )
  df_maccs_filtrado = filtrar_features(
      df_maccs,
      "MACCS KEYS",
      target_col="pChEMBL_Value",
      var_threshold=0.1,
      corr_threshold=0.80,
  )

  # Exportar datasets curados
  ruta_fisicoq = os.path.join(output_dir, "fisicoquimicos_filtrados.csv")
  ruta_maccs = os.path.join(output_dir, "maccs_filtrados.csv")

  df_fisicoq_filtrado.to_csv(ruta_fisicoq, index=False)
  df_maccs_filtrado.to_csv(ruta_maccs, index=False)

  print(f"\n-> Matriz fisicoquímica final guardada en: {ruta_fisicoq}")
  print(f"-> Matriz MACCS Keys final guardada en: {ruta_maccs}")
  print("¡Extracción y selección de características completada con éxito!")


if __name__ == "__main__":
  main()