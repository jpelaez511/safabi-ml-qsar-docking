# Descubrimiento computacional de nuevos candidatos antibacterianos frente a *Staphylococcus aureus*: integración de modelos ML-QSAR y *docking* molecular sobre la quimioteca marina CMNPD

Repositorio oficial del código y los flujos de trabajo computacionales desarrollados en el marco del **Trabajo de Fin de Máster (TFM)** en Bioinformática (curso académico 2025-2026).

| | |
|---|---|
| **Autor** | Jaime Peláez Sánchez |
| **Director del TFM** | Jose María Guijarro Blanco |
| **Titulación** | Máster Universitario en Bioinformática |
| **Institución** | Universidad Internacional de Valencia (VIU) |
| **Licencia** | [MIT](#7-licencia-y-cita) |

---

## Tabla de contenidos

1. [Contexto y justificación](#1-contexto-y-justificación)
2. [Flujo metodológico (*pipeline overview*)](#2-flujo-metodológico-pipeline-overview)
3. [Arquitectura del repositorio](#3-arquitectura-del-repositorio)
4. [Instalación y requisitos](#4-instalación-y-requisitos)
5. [Guía de ejecución paso a paso](#5-guía-de-ejecución-paso-a-paso)
6. [Compuesto cabeza de serie: enisorina E](#6-compuesto-cabeza-de-serie-enisorina-e)
7. [Licencia y cita](#7-licencia-y-cita)

---

## 1. Contexto y justificación

La rápida expansión de la resistencia antimicrobiana en los patógenos del grupo ESKAPE —con especial virulencia en las cepas nosocomiales y comunitarias de *Staphylococcus aureus* resistente a meticilina (SARM/MRSA)— plantea una amenaza crítica para la salud pública. Ante las elevadas tasas de atrición, los dilatados plazos y los costes prohibitivos asociados al cribado biológico empírico convencional, la quimioinformática emerge como una alternativa racional y escalable.

Este proyecto implementa un flujo de trabajo computacional híbrido (*ligand-based* y *structure-based*) diseñado para explorar la quimioteca marina **CMNPD** (*Comprehensive Marine Natural Products Database*). El objetivo es identificar y priorizar inhibidores moleculares selectivos frente a la enzima enoil-ACP reductasa I (**saFabI**), un punto nodal regulatorio en la síntesis de ácidos grasos bacterianos (ruta FAS-II).

---

## 2. Flujo metodológico (*pipeline overview*)

El flujo de trabajo se estructura de forma modular, automatizada y secuencial en seis scripts de Python:

```text
                          [Datos ChEMBL]
                                │
                                ▼
        01_data_curation.py ─────────────────► Curación, descarte de outliers
                                │                y estandarización química (RDKit)
                                ▼
    02_feature_extraction.py ─────────────────► Descriptores fisicoquímicos (NCATS)
                                │                y huellas MACCS Keys + filtrado
                                ▼
        03_train_qsar.py ─────────────────────► Entrenamiento, optimización
                                │                (GridSearchCV) y validación (Random Forest)
                                ▼
  04_virtual_screening.py  ◄──────────────────  [Quimioteca CMNPD (.sdf)]
                                │                Cribado masivo y consenso estricto
                                │                (pChEMBL ≥ 6.0)
                                ▼
05_druggability_and_scaffolds.py ─────────────► Filtro Ro5 (Lipinski flexible) y
                                │                diversidad estructural (Bemis-Murcko)
                                ▼
    06_prepare_ligands_3d.py ─────────────────► Generación 3D (ETKDG) y minimización
                                                 energética (MMFF94) para docking
```

---

## 3. Arquitectura del repositorio

```text
saFabI-ml-qsar-docking/
├── data/
│   ├── raw/                                # Datos originales, intactos
│   │   ├── IC50_FabI.csv                   # Registros brutos descargados de ChEMBL
│   │   └── cmnpd-07-2026.sdf               # Base de datos conformacional de CMNPD
│   └── processed/                          # Datos intermedios curados y matrices
│       ├── chembl_curado.csv
│       ├── fisicoquimicos_filtrados.csv
│       ├── maccs_filtrados.csv
│       ├── candidatos_cmnpd_fisicoquimicos.csv
│       └── candidatos_cmnpd_maccs.csv
├── models/                                 # Modelos de Machine Learning serializados
│   ├── rf_fisicoquimico.joblib
│   └── rf_maccs.joblib
├── results/                                # Tablas finales, estructuras 3D y gráficos
│   ├── figures/                            # Diagramas y gráficos analíticos (300 DPI)
│   ├── candidatos_consenso_top.csv         # Moléculas con pChEMBL consenso ≥ 6.0
│   ├── CMNPD_Lipinski_Flexible.csv         # Moléculas que superan el cribado Ro5
│   ├── CMNPD_Representantes_Elite.csv      # Moléculas líderes por familia estructural
│   └── Top_*_Pred_*.sdf                    # Confórmeros 3D minimizados
├── src/                                    # Código fuente modular (.py)
│   ├── 01_data_curation.py
│   ├── 02_feature_extraction.py
│   ├── 03_train_qsar.py
│   ├── 04_virtual_screening.py
│   ├── 05_druggability_and_scaffolds.py
│   └── 06_prepare_ligands_3d.py
├── environment.yml                         # Especificación reproducible de dependencias
├── .gitignore                              # Reglas de exclusión de Git
└── README.md                               # Documentación técnica
```

---

## 4. Instalación y requisitos

Se recomienda crear un entorno virtual aislado con `conda` o `mamba` en Linux, macOS o Windows (WSL2):

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/saFabI-ml-qsar-docking.git
cd saFabI-ml-qsar-docking

# 2. Construir y activar el entorno de trabajo
conda env create -f environment.yml
conda activate safabi-env
```

Principales paquetes y librerías utilizados:

| Paquete | Versión mínima | Función en el *pipeline* |
|---|---|---|
| Python | 3.10 | Entorno de ejecución |
| RDKit | 2023.09 | Estandarización química, huellas moleculares y modelado 3D |
| scikit-learn | 1.3 | Modelos de regresión *Random Forest* y validación cruzada |
| pandas / NumPy | — | Gestión de estructuras de datos y cálculo numérico |
| SciPy / Matplotlib / Seaborn | — | Análisis estadístico y generación de figuras |
| joblib | — | Computación en paralelo y serialización de estimadores |

---

## 5. Guía de ejecución paso a paso

Los scripts están diseñados para ejecutarse de forma secuencial desde el directorio raíz del proyecto.

### Paso 1 — Curación y estandarización química

```bash
python src/01_data_curation.py
```

- **Entrada:** `data/raw/IC50_FabI.csv`
- **Metodología:** filtra bioactividades frente a *S. aureus* con relación cuantitativa de igualdad (`=`). Aplica el protocolo de estandarización química de RDKit (limpieza general, eliminación de sales mediante `FragmentParent`, neutralización de cargas formales vía `Uncharger` y canonicalización de tautómeros). Aplica un tratamiento de duplicados, descartando mediciones atípicas (*outliers*) mediante el rango intercuartílico (1.5 × RIC) antes de calcular la media de afinidad pChEMBL.
- **Salida:** `data/processed/chembl_curado.csv` y gráficos descriptivos en `results/figures/`.

### Paso 2 — Extracción y selección de características moleculares

```bash
python src/02_feature_extraction.py
```

- **Entrada:** `data/processed/chembl_curado.csv`
- **Metodología:** calcula en paralelo 78 descriptores fisicoquímicos del subconjunto NCATS y 167 claves estructurales binarias MACCS Keys. Aplica una reducción de dimensionalidad en dos etapas: filtro de baja varianza (umbral < 0.1) y filtro de colinealidad de Pearson (r > 0.80).
- **Salida:** `data/processed/fisicoquimicos_filtrados.csv` (34 descriptores) y `data/processed/maccs_filtrados.csv` (49 claves).

### Paso 3 — Entrenamiento, optimización y validación de los modelos ML-QSAR

```bash
python src/03_train_qsar.py
```

- **Entrada:** matrices procesadas en `data/processed/`
- **Metodología:** partición reproducible *train/test* (80/20, `random_state=42`). Optimización de hiperparámetros mediante validación cruzada de 5 bloques (`GridSearchCV`, 5-fold CV) para dos estimadores *Random Forest* independientes. Evaluación de las predicciones individuales y de la combinación por consenso frente al conjunto de test, calculando R², RMSE, MAE y coeficiente de correlación de Pearson (r).
- **Salida:** modelos serializados en `models/` y diagramas de dispersión (*true vs. predicted*) en `results/figures/`.

### Paso 4 — Cribado virtual masivo sobre CMNPD y regla de consenso

```bash
python src/04_virtual_screening.py
```

- **Entrada:** `data/raw/cmnpd-07-2026.sdf`
- **Metodología:** parsea las estructuras moleculares de la quimioteca marina CMNPD, extrae los mismos descriptores optimizados en el Paso 2 y calcula las afinidades frente a saFabI con los estimadores entrenados. Aplica una regla de consenso estricta: solo se seleccionan las moléculas con predicción simultánea pChEMBL ≥ 6.0 (IC₅₀ ≤ 1 μM) en ambos modelos.
- **Salida:** `results/candidatos_consenso_top.csv` y gráfica de correlación inter-modelo en `results/figures/`.

### Paso 5 — Evaluación de biodisponibilidad oral y diversidad estructural

```bash
python src/05_druggability_and_scaffolds.py
```

- **Entrada:** `results/candidatos_consenso_top.csv`
- **Metodología:** evalúa los parámetros fisicoquímicos de la regla de cinco de Lipinski (Ro5) con un criterio flexible adaptado a la complejidad de los productos naturales marinos (≤ 1 violación permitida). Extrae los esqueletos moleculares mediante descomposición de andamios de Bemis-Murcko y prioriza el compuesto de mayor actividad predicha de cada familia estructural única (*best-in-class*).
- **Salida:** `results/CMNPD_Representantes_Elite.csv` y `results/CMNPD_Lipinski_Flexible.csv`.

### Paso 6 — Modelado conformacional y minimización energética 3D

```bash
python src/06_prepare_ligands_3d.py
```

- **Entrada:** `results/CMNPD_Representantes_Elite.csv`
- **Metodología:** selecciona el Top 5 de representantes élite, realiza la protonación e incorporación de hidrógenos explícitos, genera confórmeros tridimensionales reproducibles mediante el algoritmo ETKDG y minimiza las tensiones estéricas bajo el campo de fuerzas MMFF94. Exporta cada compuesto en archivos SDF individuales, listos para la simulación de acoplamiento molecular (*docking*).
- **Salida:** estructuras tridimensionales `results/Top_*_Pred_*.sdf` y ficha técnica del candidato cabeza de serie.

---

## 6. Compuesto cabeza de serie: enisorina E

El flujo metodológico identificó como candidato prioritario a la **enisorina E** (Top 3, identificador CMNPD29465):

| Propiedad | Valor |
|---|---|
| **SMILES canónico** | `CCC(=O)N(C)CCc1cc(Br)c(OCCCNC(=O)Cc2ccc(O)cc2)c(I)c1` |
| **Potencia biológica predicha (consenso)** | pChEMBL = 6.52 (IC₅₀ ≈ 303 nM) |
| **Origen natural** | Esponja marina *Iotrochota* cf. *iota* |

**Relevancia farmacológica y mecanística:** la enisorina E representa un andamiaje químico halogenado sin ensayos biológicos previos reportados frente a la ruta de biosíntesis de ácidos grasos bacterianos (FAS-II). En las simulaciones de acoplamiento molecular sobre el sitio catalítico de saFabI (PDB ID: [4BNF](https://www.rcsb.org/structure/4BNF)), sus sustituyentes bromoyodados y el grupo 4-hidroxifenilo establecen una red complementaria de enlaces de hidrógeno y contactos hidrofóbicos que eluden las dianas habituales de las mutaciones de resistencia conocidas.

---

## 7. Licencia y cita

Este proyecto ha sido desarrollado exclusivamente con fines académicos y de investigación científica en el marco de la Universidad Internacional de Valencia (VIU). El código fuente se distribuye bajo licencia **MIT**.

Si este repositorio te resulta útil, puedes citarlo como:

```
Peláez Sánchez J. Descubrimiento computacional de nuevos candidatos antibacterianos frente a
Staphylococcus aureus: integración de modelos ML-QSAR y docking molecular sobre la quimioteca
marina CMNPD [Trabajo de Fin de Máster]. Universidad Internacional de Valencia (VIU); 2026.
```
