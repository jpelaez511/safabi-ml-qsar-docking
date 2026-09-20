"""Módulo de curación y estandarización química de bioactividades de ChEMBL frente a saFabI (S. aureus)."""

import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
import seaborn as sns


def estandarizar_smiles(smiles: str) -> str:
  """Aplica el protocolo de estandarización química de RDKit sobre un SMILES:

  1. Limpieza general (Cleanup).
  2. Eliminación de sales y aislamiento del fragmento principal
  (FragmentParent).
  3. Neutralización de cargas eléctricas (Uncharger).
  4. Obtención del tautómero canónico (TautomerEnumerator).
  """
  try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return "Cannot_do"
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()
    uncharged_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()
    taut_mol = te.Canonicalize(uncharged_mol)
    return Chem.MolToSmiles(taut_mol)
  except Exception:
    return "Cannot_do"


def filtrar_outliers_y_promediar(grupo: pd.Series) -> float:
  """Calcula la media de afinidad pChEMBL excluyendo outliers mediante rango intercuartílico (IQR).

  Si el grupo tiene menos de 3 valores, calcula la media aritmética directa.
  """
  if len(grupo) >= 3:
    q1 = grupo.quantile(0.25)
    q3 = grupo.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    grupo_filtrado = grupo[
        (grupo >= limite_inferior) & (grupo <= limite_superior)
    ]
    if grupo_filtrado.empty:
      return float(grupo.median())
    return float(grupo_filtrado.mean())
  return float(grupo.mean())


def generar_figura_distribucion(
    df: pd.DataFrame, ruta_salida: str = "../results/figures"
):
  """Genera y guarda el gráfico de violín con la distribución de pChEMBL."""
  os.makedirs(ruta_salida, exist_ok=True)
  fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
  sns.violinplot(
      x=df["pChEMBL Value"],
      color="#3B82F6",
      inner="quartile",
      cut=0,
      linewidth=1.2,
      ax=ax,
  )
  ax.set_xlabel("Valor pChEMBL", fontsize=11, fontweight="bold")
  ax.set_title(
      "Distribución de Actividad (saFabI)", fontsize=12, fontweight="bold"
  )
  plt.tight_layout()
  plt.savefig(
      os.path.join(ruta_salida, "distribucion_pchembl_violin.png"),
      dpi=300,
      bbox_inches="tight",
  )
  plt.close()


def generar_esquema_outliers(ruta_salida: str = "../results/figures"):
  """Genera el diagrama metodológico del protocolo de tratamiento de outliers."""
  os.makedirs(ruta_salida, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
  ax.axis("off")
  fig.patch.set_facecolor("white")

  c_blue, c_light_blue = "#1E3A8A", "#EFF6FF"
  c_red, c_green, c_gray = "#DC2626", "#16A34A", "#687280"

  rect_main = patches.FancyBboxPatch(
      (0.02, 0.05),
      0.96,
      0.88,
      boxstyle="round, pad=0.02, rounding_size=0.03",
      facecolor=c_light_blue,
      edgecolor="#BFDBFE",
      lw=1.5,
  )
  ax.add_patch(rect_main)

  ax.text(
      0.5,
      0.84,
      "PROTOCOLO DE TRATAMIENTO DE OUTLIERS EN EL DATASET",
      ha="center",
      va="center",
      fontsize=12,
      fontweight="bold",
      color=c_blue,
  )
  ax.text(
      0.5,
      0.78,
      "Ejemplo de procesado sobre un registro real del dataset de ChEMBL",
      ha="center",
      va="center",
      fontsize=9.5,
      color=c_gray,
      fontstyle="italic",
  )

  mediciones = [
      ("Ensayo 1", "7.54", True),
      ("Ensayo 2", "5.02", False),
      ("Ensayo 3", "7.85", True),
      ("Ensayo 4", "7.37", True),
  ]
  x_positions = [0.08, 0.30, 0.52, 0.74]

  for i, (label, val, valido) in enumerate(mediciones):
    x = x_positions[i]
    color_box = "#FFFFFF" if valido else "#FEE2E2"
    color_border = c_blue if valido else c_red
    color_txt = c_blue if valido else c_red

    card = patches.FancyBboxPatch(
        (x, 0.42),
        0.18,
        0.26,
        boxstyle="round, pad=0.01, rounding_size=0.02",
        facecolor=color_box,
        edgecolor=color_border,
        lw=1.5,
    )
    ax.add_patch(card)
    ax.text(
        x + 0.09,
        0.62,
        label,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color_txt,
    )
    ax.text(
        x + 0.09,
        0.52,
        f"pChEMBL\n{val}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=color_txt,
    )
    tag_text = "✓ ACEPTADO" if valido else "✗ DESCARTADO\n(> 1.5 IQR)"
    tag_color = c_green if valido else c_red
    ax.text(
        x + 0.09,
        0.45,
        tag_text,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=tag_color,
    )

  ax.annotate(
      "",
      xy=(0.5, 0.31),
      xytext=(0.5, 0.39),
      arrowprops=dict(arrowstyle="->", color=c_blue, lw=2),
  )

  rect_bruto = patches.FancyBboxPatch(
      (0.15, 0.11),
      0.32,
      0.16,
      boxstyle="round, pad=0.01, rounding_size=0.02",
      facecolor="#F3F4F6",
      edgecolor="#9CA3AF",
      lw=1,
      linestyle="--",
  )
  ax.add_patch(rect_bruto)
  ax.text(
      0.31,
      0.21,
      "Media Bruta (Sin Filtrar)",
      ha="center",
      va="center",
      fontsize=8.5,
      color=c_gray,
  )
  ax.text(
      0.31,
      0.15,
      "pChEMBL = 6.95",
      ha="center",
      va="center",
      fontsize=11,
      fontweight="bold",
      color=c_gray,
  )

  rect_consenso = patches.FancyBboxPatch(
      (0.53, 0.11),
      0.32,
      0.16,
      boxstyle="round, pad=0.01, rounding_size=0.02",
      facecolor="#DCFCE7",
      edgecolor=c_green,
      lw=1.8,
  )
  ax.add_patch(rect_consenso)
  ax.text(
      0.69,
      0.21,
      "pChEMBL de Consenso (Definitivo)",
      ha="center",
      va="center",
      fontsize=9,
      fontweight="bold",
      color=c_green,
  )
  ax.text(
      0.69,
      0.15,
      "pChEMBL = 7.59",
      ha="center",
      va="center",
      fontsize=12,
      fontweight="bold",
      color=c_green,
  )

  plt.tight_layout()
  plt.savefig(
      os.path.join(ruta_salida, "Esquema_Metodologico_Outliers.png"),
      dpi=300,
      bbox_inches="tight",
  )
  plt.close()


def generar_esquema_estandarizacion(ruta_salida: str = "../results/figures"):
  """Genera el esquema visual paso a paso de la estandarización química con RDKit."""
  os.makedirs(ruta_salida, exist_ok=True)
  smiles_ejemplo = "Oc1cccc(C(=O)[O-])n1.Cl"
  mol_0 = Chem.MolFromSmiles(smiles_ejemplo)
  mol_1 = rdMolStandardize.Cleanup(mol_0)
  mol_2 = rdMolStandardize.FragmentParent(mol_1)
  uncharger = rdMolStandardize.Uncharger()
  mol_3 = uncharger.uncharge(mol_2)
  te = rdMolStandardize.TautomerEnumerator()
  mol_4 = te.Canonicalize(mol_3)

  etapas = [
      (
          mol_0,
          "1. SMILES Bruto (ChEMBL)",
          "• Presencia de sales (HCl)\n• Presencia de cargas\n• Tautómero Enol"
          " (-OH)",
      ),
      (
          mol_2,
          "2. FragmentParent",
          "• Eliminación de sales\n• Aislamiento del fragmento\n  covalente"
          " principal",
      ),
      (mol_3, "3. Uncharger", "• Neutralización de cargas"),
      (
          mol_4,
          "4. Tautómero Canónico",
          "• Transición de Enol (-OH) a\n  2-Piridona (=O / NH)",
      ),
  ]

  fig, axes = plt.subplots(
      1, 4, figsize=(17, 5.2), dpi=300, constrained_layout=True
  )
  for idx, (mol, titulo, cambios) in enumerate(etapas):
    ax = axes[idx]
    img = Draw.MolToImage(mol, size=(380, 280))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(titulo, fontsize=11.5, fontweight="bold", pad=12, color="#1E3A8A")
    ax.text(
        0.5,
        -0.12,
        cambios,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#1E293B",
        linespacing=1.5,
        bbox=dict(
            boxstyle="round, pad=0.6, rounding_size=0.03",
            facecolor="#F1F5F9" if idx > 0 else "#FEF2F2",
            edgecolor="#CBD5E1" if idx > 0 else "#FCA5A5",
            lw=1.3,
        ),
    )
    if idx < 3:
      ax.text(
          1.14,
          0.45,
          "→",
          transform=ax.transAxes,
          ha="center",
          va="center",
          fontsize=22,
          fontweight="bold",
          color="#2563EB",
      )

  plt.savefig(
      os.path.join(ruta_salida, "Esquema_Estandarizacion.png"),
      dpi=300,
      bbox_inches="tight",
  )
  plt.close()


def main():
  input_csv = "../data/raw/IC50_FabI.csv"
  output_csv = "../data/processed/chembl_curado.csv"
  figures_dir = "../results/figures"

  os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  os.makedirs(figures_dir, exist_ok=True)

  print("[1/5] Cargando dataset bruto de ChEMBL...")
  df_bruto = pd.read_csv(input_csv, sep=";")

  print("[2/5] Filtrando por organismo diana y tipo de ensayo...")
  df_filtrado = df_bruto[
      df_bruto["Assay Organism"] == "Staphylococcus aureus"
  ].copy()
  df_filtrado = df_filtrado[
      ["Smiles", "Standard Relation", "pChEMBL Value"]
  ].dropna(how="any")

  # Mantener solo relaciones cuantitativas de igualdad estricta
  df_filtrado = df_filtrado[
      df_filtrado["Standard Relation"].isin(["'='", "="])
  ].copy()

  print("[3/5] Estandarizando estructuras moleculares con RDKit...")
  df_filtrado["Smiles_Estandarizado"] = df_filtrado["Smiles"].apply(
      estandarizar_smiles
  )
  df_filtrado = df_filtrado[
      df_filtrado["Smiles_Estandarizado"] != "Cannot_do"
  ].copy()

  print("[4/5] Tratamiento de duplicados y eliminación de outliers (IQR)...")
  df_curado = (
      df_filtrado.groupby(["Smiles_Estandarizado", "Standard Relation"])[
          "pChEMBL Value"
      ]
      .apply(filtrar_outliers_y_promediar)
      .reset_index()
  )
  df_curado.rename(columns={"Smiles_Estandarizado": "Smiles"}, inplace=True)

  print(f"       -> Compuestos finales curados: {len(df_curado)}")
  df_curado.to_csv(output_csv, index=False)
  print(f"       -> Archivo exportado en: {output_csv}")

  print("[5/5] Generando figuras metodológicas del manuscrito...")
  generar_figura_distribucion(df_curado, figures_dir)
  generar_esquema_outliers(figures_dir)
  generar_esquema_estandarizacion(figures_dir)
  print("¡Curación de datos completada con éxito!")


if __name__ == "__main__":
  main()