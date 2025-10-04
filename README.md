# Sentiment Analysis Project - Restaurantes de Sigüenza

## Descripción

Este proyecto realiza un análisis de sentimiento sobre reseñas de restaurantes de Sigüenza, usando BERT en español.
Incluye tres fases:

- Fase 1: Entrenamiento inicial de un modelo BERT para clasificar reseñas como positiva o no positiva.

- Fase 2: Fine-tuning sobre el dataset específico de Sigüenza, congelando capas del modelo para evitar overfitting.

- Fase 3: Generación de un PDF de resultados por restaurante, con gráficos de distribución de sentimientos, ratings y wordclouds de elogios/críticas.


## Estructura del proyecto

´´´
sentiment_analysis_project/
│
├── data/
│   └── ResenasSiguenzaNuevo.xlsx       # Dataset de reseñas
│
├── models/
│   └── fase1/                         # Modelo fase 1
│   └── fase2/                         # Modelo fase 2 (fine-tuning)
│
├── outputs/
│   └── reporte_final_siguenza.pdf     # PDF final generado
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuración de rutas y device
│   ├── utils.py                        # Funciones auxiliares
│   ├── preprocess.py                   # Preprocesamiento y tokenización
│   ├── train_phase1.py                 # Entrenamiento Fase 1
│   ├── train_phase2.py                 # Fine-tuning Fase 2
│   ├── generate_report.py              # Generación de PDF (Fase 3)
│
├── requirements.txt                    # Librerías necesarias
├── README.md
└── run_pipeline.py                     # Script maestro para ejecutar todo
´´´

## Requisitos

Python >= 3.9

CUDA (opcional, si se desea GPU)

Instalar dependencias:

´´´
pip install -r requirements.txt
´´´

Uso
1️⃣ Entrenar modelo inicial (Fase 1)
´´´
python src/train_phase1.py
´´´

El modelo se guardará en models/fase1.

2️⃣ Fine-tuning en dataset de Sigüenza (Fase 2)
´´´
python src/train_phase2.py
´´´

El modelo se guardará en models/fase2.

3️⃣ Generar PDF con resultados (Fase 3)
´´´
python src/generate_report.py
´´´

El PDF final se guardará en outputs/reporte_final_siguenza.pdf.

4️⃣ Ejecutar todo el pipeline
´´´
python run_pipeline.py
´´´

Esto ejecuta las fases 1, 2 y 3 de manera automática.

## Estructura de los datos

El dataset debe tener las siguientes columnas:

- text: texto de la reseña

- rating: puntuación numérica (1 a 5)

- restaurant: nombre del restaurante

Opcionalmente se pueden agregar más columnas, pero text y rating son obligatorias.

## Funcionalidades

- Clasificación de reseñas en positiva / no positiva.

- Fine-tuning sobre dataset específico de la región.

- Generación de gráficos:

Distribución de sentimientos

Wordclouds de elogios y críticas

Distribución de ratings y relación rating vs sentimiento

- PDF profesional para presentar al cliente.

## Buenas prácticas implementadas

- Código modular y organizado por fases.

- Configuración centralizada en config.py.

- Semilla fija (SEED) para reproducibilidad.

- Funciones auxiliares en utils.py (limpieza, métricas, mapping).

- Tokenización y mapeo de labels robusto.

- PDF con gráficas dinámicas y wordclouds.

## Autor

Proyecto desarrollado por Maria Jose Suarez.

