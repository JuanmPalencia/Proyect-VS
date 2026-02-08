# Urban VS - Sistema Inteligente de Análisis de Tráfico Aéreo con Blockchain

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![BSV](https://img.shields.io/badge/Blockchain-BSV-EAB300.svg)](https://bitcoinsv.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NeuralHack](https://img.shields.io/badge/Hackathon-NeuralHack%202026-orange.svg)]()

**Urban VS** transforma el análisis de tráfico urbano mediante la convergencia de inteligencia artificial y blockchain. Este sistema procesa imágenes aéreas capturadas por drones o cámaras de vigilancia, detectando y analizando vehículos con un modelo YOLOv8 fine-tuned específicamente en datasets aéreos. Cada análisis queda registrado de forma inmutable en blockchain BSV, proporcionando evidencia criptográficamente verificable que resuelve problemas reales de auditoría y transparencia en gestión urbana.

---

## Capacidades del Sistema

Urban VS automatiza el ciclo completo de análisis de tráfico aéreo:

**Detección Inteligente**: Identifica 4 tipos de vehículos (coches, motos, camiones, autobuses) con 87.2% de precisión (mAP@50), superando modelos genéricos gracias al fine-tuning especializado en perspectivas aéreas.

**Análisis Avanzado**: Calcula 12+ métricas incluyendo grid de densidad 5x5, porcentaje de ocupación espacial, detección de colisiones mediante IoU, clasificación automática de riesgo en 4 niveles, y análisis diferenciado por zonas.

**Verificación Blockchain**: Cada análisis genera un hash SHA-256 que se registra en BSV mediante transacciones OP_RETURN. El timestamp del bloque certifica el momento exacto, creando evidencia inmutable auditable por cualquiera.

**Simulación Predictiva**: Permite predecir métricas bajo escenarios hipotéticos (diferentes horas, condiciones climáticas, eventos especiales) sin necesidad de imágenes reales, facilitando la planificación urbana basada en datos.

**Interfaz Completa**: Aplicación Streamlit multi-página con visualizaciones interactivas (heatmaps, grids, overlays), además de API REST para integración con sistemas externos.

---

## Características Principales

### Detección con YOLOv8 Fine-Tuned

El sistema utiliza YOLOv8, estado del arte en detección de objetos, pero con una diferencia crucial: **fine-tuning específico en datasets aéreos**. Esta especialización combina dos datasets complementarios:

- **Dataset UAV**: ~5,000 imágenes de tráfico urbano capturadas por drones con 9 escenas diferentes
- **Dataset Roundabout**: ~800 imágenes especializadas en geometría circular de rotondas

Mientras los modelos genéricos son entrenados en perspectivas terrestres, nuestro modelo domina los ángulos cenital y nadir característicos de la vigilancia aérea. Esta especialización se refleja en una precisión de **87.2% mAP@50** en validación, superando consistentemente a YOLOv8 base en escenarios de tráfico aéreo. El modelo detecta con alta confianza incluso en condiciones desafiantes como sombras pronunciadas, diferentes altitudes de captura y variaciones de iluminación.

### Análisis de Métricas Avanzadas

Más allá del conteo simple, el sistema implementa un análisis multi-dimensional del flujo de tráfico. El **grid de densidad** divide cada imagen en 25 celdas, identificando hot spots donde la concentración vehicular supera umbrales críticos. El **porcentaje de ocupación** relaciona el área total ocupada por vehículos con el espacio disponible, proporcionando una métrica intuitiva del nivel de saturación.

La **detección de colisiones** utiliza Intersection over Union (IoU) para identificar vehículos con solapamiento significativo (>0.4), clasificando severidad en tres niveles. Este análisis considera tanto la proximidad espacial como las diferencias de tamaño entre vehículos para filtrar falsos positivos.

La **clasificación automática de riesgo** integra múltiples factores: número y severidad de colisiones detectadas, densidad máxima en el grid, porcentaje de vehículos pesados, y ocupación general. El resultado es una categorización en 4 niveles (LOW/MEDIUM/HIGH/CRITICAL) que facilita la toma de decisiones operativas. Para escenas de rotondas, se aplican algoritmos especializados que consideran la geometría circular y patrones de flujo característicos.

### Blockchain para Evidencia Inmutable

La integridad criptográfica se construye desde un **payload canónico** donde los datos se serializan de forma determinística: claves JSON ordenadas alfabéticamente, tipos normalizados, sin espacios en blanco. Este payload se hashea con SHA-256, generando una huella digital única de 64 caracteres que identifica inequívocamente el análisis.

El hash se inscribe en blockchain BSV mediante transacciones con **OP_RETURN**, un tipo especial de output que permite embeber datos arbitrarios en la cadena. El **timestamp del bloque** donde se confirma la transacción certifica criptográficamente el momento del análisis, consensuado por toda la red distribuida de mineros. Modificar retroactivamente cualquier dato requeriría recomputar el proof-of-work de todos los bloques subsecuentes, lo cual es computacionalmente inviable.

La elección de **Bitcoin SV** se fundamenta en características técnicas específicas: fees ultra-bajos (~100 satoshis por transacción, equivalente a $0.00004 USD), capacidad de throughput superior a 10,000 TPS, y la ausencia de límites en el tamaño de datos OP_RETURN (otras blockchains limitan a 80 bytes). Esto permite un equilibrio óptimo entre costo operativo y robustez de la verificación distribuida.

### Simulador de Tráfico

El simulador implementa un modelo estocástico que combina factores determinísticos con variabilidad aleatoria para generar escenarios realistas sin requerir imágenes. Los **factores temporales** incluyen multiplicadores por hora del día (desde 0.05 a las 3 AM hasta 1.00 en hora pico) y reducción del 40% en fines de semana. Los **factores climáticos** modelan el impacto de condiciones adversas: lluvia reduce tráfico 20%, niebla 30%, mientras días nublados tienen efecto mínimo (-5%).

Los **eventos especiales** permiten simular concentraciones extraordinarias (conciertos, eventos deportivos) con incrementos de hasta 45% en el tráfico. Cada **tipo de escena** (urban_road, roundabout, highway) tiene capacidades base y distribuciones de clases vehiculares específicas: autopistas concentran más camiones, rotondas favorecen vehículos ligeros.

Esta herramienta responde preguntas de planificación críticas: *¿Qué ocurre si cierro un carril para obras durante hora punta? ¿Cómo afectará un evento masivo al flujo vehicular? ¿Justifica la inversión en infraestructura adicional?* El simulador proporciona métricas cuantitativas que fundamentan decisiones de gestión urbana.

---

## Instalación y Configuración

### Instalación

```bash
# 1. Clonar el repositorio
git clone <repository-url>
cd Proyect-VS

# 2. Instalar dependencias
pip install -r requirements.txt
```

Este comando instalará todas las dependencias necesarias:

- `ultralytics` (YOLOv8)
- `streamlit` (interfaz web)
- `fastapi` + `uvicorn` (API REST)
- `bsvlib` (blockchain BSV)
- `opencv-python`, `numpy`, `pandas`, `scipy` (procesamiento de imágenes y datos)

### Configuración Blockchain (Ya Lista para Usar)

**Buena noticia**: El sistema viene **preconfigurado con una clave privada BSV válida y fondeada**. Esto significa que puedes comenzar a usar el registro blockchain inmediatamente sin necesidad de:

- Crear tu propia wallet BSV
- Adquirir BSV en exchanges
- Configurar claves manualmente

La configuración en `.env` ya incluye:

```env
# Blockchain BSV (Preconfigurado - No modificar para evaluación)
BSV_NETWORK=main
BSV_PRIVATE_KEY=<clave_incluida>
ARC_URL=https://arc.gorillapool.io

# Modelo YOLO
YOLO_MODEL=runs/detect/runs/train/traffic_finetune4/weights/best.pt
CONFIDENCE_THRESHOLD=0.25
DEVICE=cpu
```

La clave incluida tiene suficiente balance para realizar **cientos de registros blockchain** sin problemas. Esto facilita la evaluación del proyecto en el hackathon sin barreras técnicas de configuración blockchain.

---

## Uso del Sistema

### Opción 1: Uso Directo de la Aplicación (Recomendado)

Esta es la forma más rápida de comenzar. El sistema incluye un **modelo YOLOv8 pre-entrenado** que ya ha sido fine-tuned en los datasets de tráfico aéreo.

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (`http://localhost:8501`).

#### Funcionalidades de la Aplicación

**1. Análisis de Imágenes**

- Sube una imagen aérea (JPG/PNG, hasta 5MB)
- Marca si es una rotonda (activa análisis especializado)
- Obtén en 5-10 segundos:
  - Detecciones con bounding boxes coloreados
  - Heatmap de densidad con escala de colores
  - Grid 5x5 mostrando conteo por zona
  - Indicadores de colisiones potenciales
  - 12+ métricas calculadas
  - Hash SHA-256 del análisis
  - Transacción blockchain con enlace al explorador

**2. Verificación de Evidencia**

- Ingresa un hash SHA-256 de un análisis previo
- Verifica su existencia en blockchain
- Consulta metadatos completos (timestamp, métricas, txid)
- Accede al explorador blockchain para ver la transacción confirmada

**3. Simulador** 

- Configura escenario: fecha/hora, tipo de escena, clima, eventos
- Ajusta cantidad de vehículos por tipo (sliders interactivos)
- Observa métricas en tiempo real
- Compara diferentes escenarios para planificación

**4. Registro de Evidencia**

- Lista todos los análisis históricos
- Filtra por dataset, ordena por fecha
- Exporta resultados a JSON
- Busca por scene_id o análisis específicos

---

### Opción 2: Entrenar Tu Propio Modelo (Avanzado)

Si deseas experimentar con el entrenamiento o usar tus propios datasets:

#### Nota sobre los Datasets

**La carpeta `data/` contiene únicamente una pequeña muestra (~50 imágenes) para demostración.** Estas muestras son suficientes para:

- Ejecutar la aplicación y ver todas las funcionalidades
- Probar los análisis en imágenes de ejemplo
- Verificar que el sistema funciona correctamente

**Para entrenar el modelo**, necesitas descargar los datasets completos desde Kaggle:

1. **traffic_aerial_images_for_vehicle_detection/**

   - ~5,000 imágenes de tráfico urbano por drones
   - Extraer en `data/traffic_aerial_images_for_vehicle_detection/`
2. **roundabout_aerial_images_for_vehicle_detection/**

   - ~800 imágenes especializadas en rotondas
   - Extraer en `data/roundabout_aerial_images_for_vehicle_detection/`

**¡Importante!** Respetar exactamente estos nombres de carpeta para que `setup_train.py` funcione correctamente.

#### Pasos de Entrenamiento

```bash
# 1. Preparar dataset unificado (combina UAV + Roundabout)
python setup_train.py
```

Este script automáticamente:

- Convierte el dataset Roundabout de CSV a formato YOLO
- Unifica las clases (mapea "cycle" → "motorcycle")
- Combina ~5,800 imágenes totales
- Hace split 85% train / 15% val
- Genera `data/combined/data.yaml` con configuración

```bash
# 2. Entrenar modelo
python train.py
```

Parámetros opcionales:

```bash
python train.py --epochs 100 --batch 16 --device cuda
```

El entrenamiento toma 2-8 horas dependiendo del hardware. El modelo se guarda en `runs/detect/runs/train/traffic_finetune_N/weights/best.pt`.

---

### Uso del CLI (Procesamiento Batch)

Para automatización o integración con scripts:

```bash
# Analizar una imagen
python cli.py analyze path/to/image.jpg --register

# Analizar directorio completo
python cli.py analyze data/samples/ --output results.json

# Verificar evidencia por hash
python cli.py verify e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

---

## API REST (Desarrollo Futuro)

> **Nota**: La API REST está completamente implementada en el código (`api.py`) y es funcional. Sin embargo, se presenta como funcionalidad de desarrollo futuro para mantener el foco en la demostración de la interfaz Streamlit durante la evaluación del hackathon.

La API REST con FastAPI está diseñada para facilitar la **integración de Urban VS con sistemas externos**:

### Casos de Uso de la API

- **Aplicaciones móviles** (iOS/Android) que capturen fotos de tráfico y envíen a Urban VS
- **Sistemas municipales de gestión** (SCADA, control de semáforos)
- **Automatización con scripts** Python/Node.js/Java para procesamiento programático
- **Arquitectura de microservicios** donde Urban VS actúa como servicio de detección
- **Webhooks y notificaciones** cuando se detecten condiciones críticas

### Endpoints Principales

| Endpoint                  | Método | Descripción                                                                       |
| ------------------------- | ------- | ---------------------------------------------------------------------------------- |
| `/analyze`              | POST    | Analiza imagen (multipart/form-data). Retorna detecciones, métricas, hash y tx_id |
| `/verify?hash=<sha256>` | GET     | Busca análisis por hash. Retorna evidence_record completo o null                  |
| `/records?limit=50`     | GET     | Lista los N registros más recientes del ledger                                    |
| `/simulate`             | POST    | Ejecuta simulación what-if con JSON de parámetros                                |
| `/health`               | GET     | Health check para monitoreo (retorna estado del modelo y blockchain)               |

### Ejecución de la API (Cuando se Implemente en Producción)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

Documentación interactiva disponible en:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Ejemplo de Uso Programático

```python
import requests

# Analizar imagen vía API
with open("traffic_scene.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"image": f},
        params={"dataset_id": "api_test", "is_roundabout": False}
    )

result = response.json()
print(f"Detectados {result['metrics']['total_vehicles']} vehículos")
print(f"Nivel de riesgo: {result['metrics']['risk_level']}")
print(f"TX Blockchain: {result['tx_id']}")
```

---

## Arquitectura del Sistema

Urban VS sigue una **arquitectura modular de capas** donde cada componente tiene responsabilidades claramente definidas:

```
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND (Streamlit)                           │
│  Interfaz visual para análisis, simulación y verificación   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                  API REST (FastAPI) - Futuro                │
│  Endpoints: /analyze, /verify, /simulate, /records, /health│
└────┬───────────────────────────────────────────────┬────────┘
     │                                               │
┌────┴─────────────┐                    ┌───────────┴─────────┐
│   Detection      │                    │   Metrics Analyzer  │
│   (YOLOv8)       │                    │   (12+ métricas)    │
│                  │                    │                     │
│ - Fine-tuned en  │                    │ - Grid densidad     │
│   UAV + Roundab. │                    │ - Ocupación zonas   │
│ - mAP: 87.2%     │                    │ - Detección colision│
│ - 4 clases       │                    │ - Clasificac. riesgo│
└────────┬─────────┘                    └───────────┬─────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            │
               ┌────────────┴─────────────┐
               │  Hashing & Integrity     │
               │  (SHA-256 canónico)      │
               │                          │
               │ - JSON ordenado          │
               │ - Tipos normalizados     │
               │ - Reproducible           │
               └────────────┬─────────────┘
                            │
               ┌────────────┴─────────────┐
               │  Blockchain Adapter      │
               │  (BSV + Local Ledger)    │
               │                          │
               │ - OP_RETURN tx           │
               │ - ARC broadcast          │
               │ - WhatsOnChain queries   │
               │ - Ledger JSONL local     │
               └──────────────────────────┘
```

### Componentes Detallados

**src/detection/**: Detector de vehículos con YOLOv8. Implementa fusión de modelo principal + fallback COCO para clases escasas (buses).

**src/metrics/**: Analizador de métricas de tráfico. Calcula grid de densidad, ocupación por zonas, detección de colisiones mediante IoU, clasificación de riesgo multi-factorial.

**src/hashing/**: Sistema de integridad criptográfica. Genera JSON canónico (claves ordenadas, sin espacios), aplica SHA-256, maneja tipos numpy.

**src/blockchain/**: Adaptador dual (BSV on-chain + ledger local). Construye transacciones OP_RETURN con bsv-sdk, transmite vía ARC, verifica con WhatsOnChain.

**src/visualization/**: Generación de overlays visuales. Bounding boxes, heatmaps gaussianos, grids de densidad, indicadores de colisión.

**src/simulator/**: Motor de simulación what-if. Modelo estocástico con 15+ factores (hora, clima, eventos), distribución realista de vehículos.

---

## Datasets y Entrenamiento

### Datasets Utilizados

**1. UAV Traffic Dataset (~5,000 imágenes)**

- Capturas aéreas de tráfico urbano desde drones
- Altitudes variables (50-150 metros)
- 9 escenas diferentes con diversas condiciones
- Ya viene con anotaciones YOLO
- **Clases**: car, motorcycle

**2. Roundabout Aerial Dataset (~800 imágenes)**

- Especializado en rotondas
- Resolución 1920x1080
- Anotaciones en CSV (convertidas a YOLO automáticamente)
- **Clases**: car, cycle, truck, bus

**3. Dataset Combinado (Generado por setup_train.py)**

- Unificación de ambos datasets
- ~5,800 imágenes totales
- 4 clases finales: car, motorcycle, truck, bus
- Split 85/15 train/val

### Métricas del Modelo

Resultados en validación (traffic_finetune4):

| Métrica                          | Valor |
| --------------------------------- | ----- |
| **mAP@50**                  | 87.2% |
| **mAP@50-95**               | 65.4% |
| **Precisión (car)**        | 91%   |
| **Precisión (motorcycle)** | 82%   |
| **Precisión (truck)**      | 88%   |
| **Precisión (bus)**        | 79%   |

---

## Blockchain y Verificación

### ¿Por Qué Blockchain?

La integración blockchain no es cosmética - resuelve problemas reales:

**1. Auditoría Forense**: En disputas legales (accidentes, seguros), las partes pueden verificar que los datos de tráfico no han sido manipulados.

**2. Transparencia Municipal**: Gobiernos pueden publicar análisis de tráfico verificables por ciudadanos, generando confianza.

**3. Cobro por Congestión**: Sistemas de tarifas donde los conductores pueden auditar que los cálculos son correctos.

**4. Cadena de Custodia**: El timestamp del bloque certifica cuándo ocurrió el análisis (importante para evidencia temporal).

### Flujo de Registro

1. **Análisis** → Se detectan vehículos y calculan métricas
2. **Payload Canónico** → JSON con campos ordenados alfabéticamente
3. **Hash SHA-256** → Huella digital única de 64 caracteres hex
4. **Transacción BSV** → Se construye TX con OP_RETURN conteniendo el hash
5. **Broadcast** → Se envía a red BSV via ARC (gorillapool.io)
6. **Confirmación** → Mineros incluyen TX en bloque (~10 min)
7. **Verificación** → Cualquiera puede consultar la TX en explorador blockchain

### Estructura de Transacción

```
Input  → Gasta UTXO previo (firmado con clave privada)
Output 0 → OP_RETURN con datos: [APP_PREFIX, hash, scene_id, version]
Output 1 → P2PKH de cambio (devuelve satoshis - fee)
Fee    → ~100-150 satoshis (~$0.00004 USD)
```

---

## Documentación Técnica

**Documentación completa disponible en**: [`DOCUMENTACION_TECNICA.docx`](DOCUMENTACION_TECNICA.docx)

Este documento Word de ~15 páginas incluye:

✅ Visión general del proyecto con contexto y motivación
✅ Arquitectura detallada (pipeline de procesamiento, módulos)
✅ Instrucciones completas de instalación y configuración
✅ Guías paso a paso para ambos modos de uso
✅ Detalles de la API REST y endpoints
✅ Explicaciones profundas de algoritmos y métricas
✅ Arquitectura de blockchain e integridad criptográfica
✅ Documentación del simulador what-if
✅ **Roadmap completo de desarrollo futuro**

---

## Desarrollo Futuro

El proyecto evoluciona en tres horizontes temporales con objetivos claros:

**Corto Plazo**: Despliegue de la API REST en infraestructura cloud con autenticación robusta y monitoring avanzado. Implementación de tracking multi-frame mediante algoritmos como DeepSORT para analizar flujos de video completos, calculando velocidades y tiempos de tránsito por zona. Optimización de performance mediante batch processing, cuantización de modelos (FP16/INT8) y cachés inteligentes, con objetivo de reducir el tiempo de inferencia de 5 a <2 segundos por imagen.

**Medio Plazo**: Expansión del catálogo de clases detectadas incluyendo bicicletas, scooters eléctricos y vehículos de emergencia. Desarrollo de modelos predictivos (Random Forest, XGBoost) que combinen métricas actuales con históricos de tráfico, datos climáticos en tiempo real y calendarios de eventos. Integración con sistemas municipales de control de semáforos para optimización adaptativa basada en análisis real.

**Largo Plazo**: Capacidades avanzadas como detección de comportamientos anómalos mediante LSTM autoencoders (vehículos en sentido contrario, peatones en autopistas), predicción de tráfico futuro con Transformers (30 minutos hacia adelante, MAPE <15%), y sistemas de cobro por congestión verificables on-chain. Expansión a múltiples blockchains (Ethereum, Polygon, Hyperledger) según casos de uso, desarrollo de aplicación móvil para participación ciudadana, y evolución hacia modelos 3D-aware para estimación precisa de pose vehicular.

---

## Casos de Uso y Aplicaciones

### Gestión Municipal

- Monitoreo de flujo de tráfico en tiempo real
- Optimización de semáforos basada en datos
- Identificación de puntos calientes de congestión
- Planificación de infraestructura (nuevos carriles, rotondas)

### Seguridad y Emergencias

- Detección rápida de accidentes
- Evaluación de riesgo en tiempo real
- Priorización de despacho de recursos
- Análisis forense con evidencia verificable

### Investigación y Desarrollo

- Dataset público con hashes verificables
- Papers académicos reproducibles
- Benchmark de algoritmos de detección
- Estudios de movilidad urbana

### Sector Privado

- Sistemas de seguros con evidencia tamper-proof
- Cobro por uso de infraestructura
- Análisis de impacto de eventos comerciales
- Optimización de rutas de flotas

---

## Tecnologías Utilizadas

| Categoría               | Tecnología          | Propósito                              |
| ------------------------ | -------------------- | --------------------------------------- |
| **ML/CV**          | YOLOv8 (Ultralytics) | Detección de objetos state-of-the-art  |
| **Blockchain**     | Bitcoin SV + bsvlib  | Registro inmutable on-chain             |
| **Backend**        | FastAPI + Uvicorn    | API REST asíncrona de alto rendimiento |
| **Frontend**       | Streamlit            | Interfaz web interactiva                |
| **Hashing**        | SHA-256 (hashlib)    | Integridad criptográfica               |
| **Visión**        | OpenCV               | Procesamiento de imágenes              |
| **Datos**          | NumPy, Pandas        | Manipulación de arrays y DataFrames    |
| **Visualización** | Matplotlib, Scipy    | Heatmaps y gráficos                    |

---

## Licencia

Este proyecto ha sido desarrollado para **NeuralHack 2026**.

---

**Desarrollado para NeuralHack 2026**
*Convergencia de IA y Blockchain para Ciudades Inteligentes*
