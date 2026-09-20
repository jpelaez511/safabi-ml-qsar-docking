"""
Módulo de entrenamiento, optimización y validación de modelos Random Forest (ML-QSAR)
frente a la enoil-ACP reductasa saFabI de Staphylococcus aureus.
"""

import os
from typing import Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Cuadrícula de hiperparámetros para la búsqueda en validación cruzada
PARAM_GRID = {
    'n_estimators': [200, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_depth': [None, 10, 15],
    'bootstrap': [True],
}


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
  """Calcula R2, RMSE, MAE y coeficiente de correlación de Pearson (r)."""
  r2 = r2_score(y_true, y_pred)
  rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
  mae = float(mean_absolute_error(y_true, y_pred))
  r, _ = pearsonr(y_true, y_pred)
  return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Pearson_r': r}


def optimizar_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    nombre_modelo: str,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> RandomForestRegressor:
  """Ejecuta GridSearchCV con 5-fold CV y devuelve el mejor estimador optimizado."""
  print(f"\n[Optimizando] Grid Search CV para modelo {nombre_modelo}...")
  rf_base = RandomForestRegressor(random_state=42)
  grid = GridSearchCV(
      estimator=rf_base,
      param_grid=PARAM_GRID,
      cv=cv_folds,
      n_jobs=n_jobs,
      scoring='r2',
      verbose=1,
  )
  grid.fit(x_train, y_train)

  print(f"Mejores hiperparámetros ({nombre_modelo}): {grid.best_params_}")
  print(f"R2 medio en validación cruzada (Train): {grid.best_score_:.3f}")
  return grid.best_estimator_


def generar_grafico_dispersion(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    titulo: str,
    nombre_archivo: str,
    metricas: Dict[str, float],
    output_dir: str = '../results/figures',
):
  """Genera y exporta el gráfico de dispersión 'Real vs. Predicho' en alta resolución."""
  os.makedirs(output_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

  df_plot = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
  sns.scatterplot(
      data=df_plot,
      x='True',
      y='Predicted',
      alpha=0.8,
      color='#1E40AF',
      edgecolor='black',
      s=45,
      ax=ax,
  )

  # Línea diagonal ideal y = x
  lims = [
      min(y_test.min(), y_pred.min()) - 0.3,
      max(y_test.max(), y_pred.max()) + 0.3,
  ]
  ax.plot(lims, lims, '--', color='#EF4444', linewidth=1.5, label='Ideal (y = x)')

  ax.set_xlim(lims)
  ax.set_ylim(lims)
  ax.set_xlabel(r'Actividad Experimental ($p\mathrm{ChEMBL}$)', fontsize=11, fontweight='bold')
  ax.set_ylabel(r'Predicción del Modelo ($p\mathrm{ChEMBL}$)', fontsize=11, fontweight='bold')
  ax.set_title(titulo, fontsize=12, fontweight='bold', pad=10)

  # Cuadro de texto con métricas
  texto_stats = (
      f"$R^2$ = {metricas['R2']:.3f}\n"
      f"RMSE = {metricas['RMSE']:.3f}\n"
      f"Pearson $r$ = {metricas['Pearson_r']:.3f}"
  )
  ax.text(
      0.05,
      0.92,
      texto_stats,
      transform=ax.transAxes,
      fontsize=9.5,
      verticalalignment='top',
      bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2),
  )

  ax.legend(loc='lower right', frameon=True)
  plt.tight_layout()
  ruta_salida = os.path.join(output_dir, nombre_archivo)
  plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
  plt.close()


def main():
  # Definición de rutas relativas
  input_fisiq = '../data/processed/fisicoquimicos_filtrados.csv'
  input_maccs = '../data/processed/maccs_filtrados.csv'
  models_dir = '../models'
  figures_dir = '../results/figures'

  os.makedirs(models_dir, exist_ok=True)
  os.makedirs(figures_dir, exist_ok=True)

  print('[1/5] Cargando matrices de características curadas...')
  df_fisico = pd.read_csv(input_fisiq)
  df_maccs = pd.read_csv(input_maccs)

  y = df_maccs['pChEMBL_Value'].values
  x_fisico = df_fisico.drop(columns=['pChEMBL_Value']).values
  x_maccs = df_maccs.drop(columns=['pChEMBL_Value']).values

  print('[2/5] Generando partición Train/Test (80/20, random_state=42)...')
  (
      x_train_fisiq,
      x_test_fisiq,
      y_train,
      y_test,
  ) = train_test_split(x_fisico, y, test_size=0.2, random_state=42)

  (
      x_train_mac,
      x_test_mac,
      _,
      _,
  ) = train_test_split(x_maccs, y, test_size=0.2, random_state=42)

  print('[3/5] Ajuste y optimización de hiperparámetros por GridSearchCV...')
  rf_fisico_opt = optimizar_random_forest(
      x_train_fisiq, y_train, 'Fisicoquímico'
  )
  rf_maccs_opt = optimizar_random_forest(x_train_mac, y_train, 'MACCS Keys')

  print('[4/5] Evaluación en el conjunto de prueba (Test Set)...')
  y_pred_fisiq = rf_fisico_opt.predict(x_test_fisiq)
  y_pred_maccs = rf_maccs_opt.predict(x_test_mac)
  y_pred_consenso = (y_pred_fisiq + y_pred_maccs) / 2.0

  met_fisiq = calcular_metricas(y_test, y_pred_fisiq)
  met_maccs = calcular_metricas(y_test, y_pred_maccs)
  met_consenso = calcular_metricas(y_test, y_pred_consenso)

  print('\n' + '=' * 55)
  print('RESULTADOS EN CONJUNTO DE PRUEBA (TEST SET - 20%)')
  print('=' * 55)
  print(
      f"Modelo Fisicoquímico: R2 = {met_fisiq['R2']:.3f} | RMSE ="
      f" {met_fisiq['RMSE']:.3f} | Pearson r = {met_fisiq['Pearson_r']:.3f}"
  )
  print(
      f"Modelo MACCS Keys:     R2 = {met_maccs['R2']:.3f} | RMSE ="
      f" {met_maccs['RMSE']:.3f} | Pearson r = {met_maccs['Pearson_r']:.3f}"
  )
  print(
      f"Modelo Consenso:       R2 = {met_consenso['R2']:.3f} | RMSE ="
      f" {met_consenso['RMSE']:.3f} | Pearson r ="
      f" {met_consenso['Pearson_r']:.3f}"
  )
  print('=' * 55)

  print('\n[5/5] Exportando figuras de dispersión y modelos serializados...')
  generar_grafico_dispersion(
      y_test,
      y_pred_fisiq,
      'Random Forest: Descriptores Fisicoquímicos',
      'dispersion_fisicoquimico.png',
      met_fisiq,
      figures_dir,
  )
  generar_grafico_dispersion(
      y_test,
      y_pred_maccs,
      'Random Forest: MACCS Keys',
      'dispersion_maccs.png',
      met_maccs,
      figures_dir,
  )
  generar_grafico_dispersion(
      y_test,
      y_pred_consenso,
      'Modelo de Consenso (Fisicoquímico + MACCS)',
      'dispersion_consenso.png',
      met_consenso,
      figures_dir,
  )

  # Guardar los modelos entrenados
  ruta_rf_fisiq = os.path.join(models_dir, 'rf_fisicoquimico.joblib')
  ruta_rf_maccs = os.path.join(models_dir, 'rf_maccs.joblib')
  joblib.dump(rf_fisico_opt, ruta_rf_fisiq)
  joblib.dump(rf_maccs_opt, ruta_rf_maccs)

  print(f'-> Modelo fisicoquímico guardado en: {ruta_rf_fisiq}')
  print(f'-> Modelo MACCS Keys guardado en:     {ruta_rf_maccs}')
  print('¡Entrenamiento y validación completados con éxito!')


if __name__ == '__main__':
  main()