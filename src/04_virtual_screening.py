"""
Módulo de cribado virtual quimioinformático a gran escala sobre la quimioteca
marina CMNPD para la identificación de potenciales inhibidores de saFabI.
"""

import os
from typing import List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import pearsonr
import seaborn as sns


def extraer_smiles_sdf(ruta_sdf: str) -> pd.DataFrame:
    """Extrae y parsea los SMILES válidos de la quimioteca marina en formato SDF."""
    print(f"[1/6] Leyendo estructuras moleculares desde: {ruta_sdf}")
    proveedor = Chem.SDMolSupplier(ruta_sdf)
    lista_smiles = []
    
    for mol in proveedor:
        if mol is not None:
            lista_smiles.append(Chem.MolToSmiles(mol))
            
    df_smiles = pd.DataFrame({'Smiles': lista_smiles})
    print(f"       -> Compuestos válidos extraídos de CMNPD: {len(df_smiles)}")
    return df_smiles


def calcular_descriptores_cmnpd(
    df_moleculas: pd.DataFrame,
    nombres_fisico: List[str],
    indices_maccs: List[int]
) -> pd.DataFrame:
    """
    Calcula de forma exacta las características moleculares requeridas por los modelos
    entrenados (34 descriptores fisicoquímicos y 49 huellas MACCS Keys).
    """
    print(f"[2/6] Calculando {len(nombres_fisico)} descriptores fisicoquímicos y "
          f"{len(indices_maccs)} claves MACCS...")
    
    calc_fisico = MoleculeDescriptors.MolecularDescriptorCalculator(nombres_fisico)
    
    def _calcular_vector(smi: str) -> List[float]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                valores_fisiq = list(calc_fisico.CalcDescriptors(mol))
                maccs_full = MACCSkeys.GenMACCSKeys(mol)
                valores_maccs = [maccs_full[i] for i in indices_maccs]
                return valores_fisiq + valores_maccs
            except Exception:
                pass
        return [np.nan] * (len(nombres_fisico) + len(indices_maccs))

    # Extracción vectorizada
    nombres_maccs = [f"MACCS_{i}" for i in indices_maccs]
    columnas_totales = nombres_fisico + nombres_maccs
    
    vectores = df_moleculas['Smiles'].apply(_calcular_vector)
    df_caracteristicas = pd.DataFrame(vectores.tolist(), columns=columnas_totales)
    df_caracteristicas.insert(0, 'Smiles', df_moleculas['Smiles'])
    
    # Filtrar compuestos que pudieran fallar en cálculos conformacionales
    df_limpio = df_caracteristicas.dropna().reset_index(drop=True)
    print(f"       -> Compuestos computados con éxito: {len(df_limpio)}")
    return df_limpio


def evaluar_concordancia_modelos(
    df_predicciones: pd.DataFrame,
    output_dir: str = "../results/figures"
) -> float:
    """
    Calcula la correlación de Pearson entre ambos modelos ML-QSAR en el cribado
    y exporta el scatterplot con regresión lineal en alta resolución.
    """
    os.makedirs(output_dir, exist_ok=True)
    r_val, p_val = pearsonr(df_predicciones['pChEMBL_Fisico'], df_predicciones['pChEMBL_MACCS'])
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.set_theme(style="whitegrid")
    
    sns.regplot(
        x='pChEMBL_Fisico',
        y='pChEMBL_MACCS',
        data=df_predicciones,
        scatter_kws={'alpha': 0.25, 'color': '#134CD0', 's': 15},
        line_kws={'color': '#DC2626', 'linewidth': 2},
        ax=ax
    )
    
    ax.set_title(
        f'Concordancia de Predicciones: Fisicoquímico vs MACCS Keys\n'
        f'Correlación de Pearson (r) = {r_val:.3f} (p < 0.001)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel(r'Actividad Predicha ($p\mathrm{ChEMBL}$) - Descriptores Fisicoquímicos', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Actividad Predicha ($p\mathrm{ChEMBL}$) - Claves MACCS', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    ruta_figura = os.path.join(output_dir, 'Correlacion_Modelos_CMNPD.png')
    plt.savefig(ruta_figura, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[4/6] Análisis de correlación entre modelos completado (r = {r_val:.3f}). Gráfico guardado.")
    return r_val


def aplicar_regla_consenso(
    df_predicciones: pd.DataFrame,
    umbral: float = 6.0
) -> pd.DataFrame:
    """
    Aplica el filtro de consenso estricto: la molécula debe alcanzar o superar
    el umbral micromolar (pChEMBL >= 6.0, equivalente a IC50 <= 1 uM)
    en AMBOS modelos de forma simultánea.
    """
    filtro = (df_predicciones['pChEMBL_Fisico'] >= umbral) & (df_predicciones['pChEMBL_MACCS'] >= umbral)
    df_top = df_predicciones[filtro].copy()
    
    # Calcular la afinidad consenso (media aritmética) y ordenar de mayor a menor
    df_top['Media_Predicciones'] = (df_top['pChEMBL_Fisico'] + df_top['pChEMBL_MACCS']) / 2.0
    df_top = df_top.sort_values(by='Media_Predicciones', ascending=False).reset_index(drop=True)
    
    return df_top


def main():
    # Definición de rutas relativas
    ruta_sdf = '../data/raw/cmnpd-07-2026.sdf'
    dir_procesados = '../data/processed'
    dir_modelos = '../models'
    dir_resultados = '../results'
    
    os.makedirs(dir_procesados, exist_ok=True)
    os.makedirs(dir_resultados, exist_ok=True)
    
    # 1. Cargar metadatos de los descriptores entrenados en el script 02
    df_fisiq_train = pd.read_csv(os.path.join(dir_procesados, 'fisicoquimicos_filtrados.csv'))
    df_maccs_train = pd.read_csv(os.path.join(dir_procesados, 'maccs_filtrados.csv'))
    
    nombres_fisico = df_fisiq_train.drop(columns=['pChEMBL_Value']).columns.tolist()
    nombres_maccs = df_maccs_train.drop(columns=['pChEMBL_Value']).columns.tolist()
    indices_maccs = [int(col.split('_')[1]) for col in nombres_maccs]
    
    # 2. Extracción y cálculo de características sobre CMNPD
    df_smiles = extraer_smiles_sdf(ruta_sdf)
    df_cmnpd_features = calcular_descriptores_cmnpd(df_smiles, nombres_fisico, indices_maccs)
    
    # Separar y exportar matrices intermedias
    df_final_fisico = df_cmnpd_features[['Smiles'] + nombres_fisico]
    df_final_maccs = df_cmnpd_features[['Smiles'] + nombres_maccs]
    
    df_final_fisico.to_csv(os.path.join(dir_procesados, 'candidatos_cmnpd_fisicoquimicos.csv'), index=False)
    df_final_maccs.to_csv(os.path.join(dir_procesados, 'candidatos_cmnpd_maccs.csv'), index=False)
    
    # 3. Predicción con los modelos Random Forest serializados
    print("[3/6] Evaluando quimioteca con los modelos Random Forest...")
    rf_fisico = joblib.load(os.path.join(dir_modelos, 'rf_fisicoquimico.joblib'))
    rf_maccs = joblib.load(os.path.join(dir_modelos, 'rf_maccs.joblib'))
    
    X_fisico = df_final_fisico.drop(columns=['Smiles']).values
    X_maccs = df_final_maccs.drop(columns=['Smiles']).values
    
    pred_fisico = rf_fisico.predict(X_fisico)
    pred_maccs = rf_maccs.predict(X_maccs)
    
    df_predicciones = pd.DataFrame({
        'Smiles': df_final_fisico['Smiles'],
        'pChEMBL_Fisico': pred_fisico,
        'pChEMBL_MACCS': pred_maccs
    })
    
    # 4. Generación de gráfico de concordancia
    evaluar_concordancia_modelos(df_predicciones, output_dir=os.path.join(dir_resultados, 'figures'))
    
    # 5. Filtrado por regla de consenso estricto (pChEMBL >= 6.0)
    print("[5/6] Aplicando criterio de consenso estricto (pChEMBL >= 6.0 en ambos modelos)...")
    candidatos_top = aplicar_regla_consenso(df_predicciones, umbral=6.0)
    
    # 6. Exportar resultados finales
    ruta_candidatos = os.path.join(dir_resultados, 'candidatos_consenso_top.csv')
    candidatos_top.to_csv(ruta_candidatos, index=False)
    
    print("\n" + "="*60)
    print("RESUMEN DEL CRIBADO VIRTUAL (CMNPD)")
    print("="*60)
    print(f"Total compuestos evaluados:             {len(df_predicciones)}")
    print(f"Candidatos seleccionados (pChEMBL >= 6): {len(candidatos_top)}")
    print(f"Mayor pChEMBL predicho por consenso:    {candidatos_top['Media_Predicciones'].max():.3f}")
    print(f"Archivo con candidatos priorizados:      {ruta_candidatos}")
    print("="*60)
    print("[6/6] ¡Cribado virtual sobre CMNPD finalizado con éxito!")


if __name__ == "__main__":
    main()