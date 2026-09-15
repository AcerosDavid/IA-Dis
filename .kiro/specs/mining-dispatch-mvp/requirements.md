# Requirements Document

## Introduction

Este documento define los requerimientos del **Módulo 1 – MVP de Detección de Maquinaria** del sistema de despacho inteligente para operaciones mineras. El objetivo es construir una prueba de concepto funcional que cargue un video local, detecte camiones y palas mediante YOLO, y registre los eventos detectados en una base de datos SQLite local. No se contempla interfaz de usuario completa, integración con sistemas externos ni procesamiento en tiempo real de múltiples cámaras.

---

## Glossary

- **Sistema**: El sistema de detección de maquinaria minera (módulo 1 del MVP).
- **Detector**: Componente responsable de ejecutar el modelo YOLO sobre los fotogramas del video.
- **Repositorio_Eventos**: Componente responsable de persistir los eventos detectados en la base de datos SQLite.
- **API**: Interfaz FastAPI básica que expone los endpoints de consulta de eventos.
- **Evento**: Registro de la detección de una máquina en un fotograma, compuesto por: timestamp, tipo de máquina, clase/ID y descripción del evento.
- **Máquina**: Entidad física detectada; puede ser de tipo `camion` o `pala`.
- **Video_Fuente**: Archivo de video local en formato compatible con OpenCV (`.mp4`, `.avi`, `.mov`).
- **Fotograma**: Imagen individual extraída del Video_Fuente durante el procesamiento.
- **Confianza**: Valor numérico en el rango `[0.0, 1.0]` que indica la certeza de la detección del Detector.
- **Umbral_Confianza**: Valor mínimo de Confianza aceptado para registrar una detección como Evento válido.
- **SQLite_DB**: Archivo de base de datos SQLite local que almacena los Eventos.

---

## Requirements

### Requerimiento 1: Carga y validación del video fuente

**User Story:** Como desarrollador, quiero que el Sistema valide y cargue el Video_Fuente antes de iniciar el procesamiento, para garantizar que el archivo es accesible y tiene un formato compatible.

#### Criterios de Aceptación

1. WHEN se proporciona la ruta del Video_Fuente, THE Sistema SHALL verificar que el archivo existe en el sistema de archivos antes de intentar abrirlo.
2. WHEN el archivo no existe en la ruta proporcionada, THE Sistema SHALL registrar un error descriptivo en consola con la ruta recibida y detener la ejecución.
3. WHEN el archivo existe pero OpenCV no puede decodificarlo, THE Sistema SHALL registrar un error descriptivo en consola indicando el formato incompatible y detener la ejecución.
4. WHEN el Video_Fuente es cargado exitosamente, THE Sistema SHALL registrar en consola la resolución, el frame rate y el número total de fotogramas del video.

---

### Requerimiento 2: Inicialización del modelo de detección

**User Story:** Como desarrollador, quiero que el Detector cargue el modelo YOLO al inicio de la aplicación, para que el modelo esté listo antes de procesar cualquier fotograma.

#### Criterios de Aceptación

1. THE Detector SHALL cargar el modelo YOLO desde una ruta de archivo configurable mediante una variable de entorno o archivo de configuración.
2. WHEN la ruta del modelo no existe o el archivo está corrupto, THE Detector SHALL registrar un error descriptivo en consola y detener la ejecución.
3. WHEN el modelo es cargado exitosamente, THE Detector SHALL registrar en consola el nombre del modelo y el dispositivo de inferencia utilizado (`cpu` o `cuda`).
4. THE Detector SHALL aceptar un Umbral_Confianza configurable con un valor predeterminado de `0.5`.

---

### Requerimiento 3: Procesamiento de fotogramas y detección de maquinaria

**User Story:** Como desarrollador, quiero que el Detector analice cada fotograma del Video_Fuente e identifique camiones y palas, para generar eventos de detección.

#### Criterios de Aceptación

1. WHEN el Video_Fuente es cargado, THE Detector SHALL procesar cada Fotograma de forma secuencial desde el primer hasta el último fotograma.
2. WHEN el Detector analiza un Fotograma, THE Detector SHALL ejecutar la inferencia YOLO y retornar únicamente las detecciones con clase `camion` o `pala`.
3. WHEN una detección tiene un valor de Confianza menor al Umbral_Confianza, THE Detector SHALL descartar esa detección sin registrar un Evento.
4. WHEN una detección supera el Umbral_Confianza, THE Detector SHALL generar un Evento con los campos: `timestamp` del fotograma, `tipo_maquina`, `clase_id`, `confianza` y `numero_fotograma`.
5. WHEN el Detector finaliza el procesamiento de todos los fotogramas, THE Detector SHALL registrar en consola el número total de fotogramas procesados y el número total de Eventos generados.

---

### Requerimiento 4: Persistencia de eventos en SQLite

**User Story:** Como desarrollador, quiero que el Repositorio_Eventos almacene cada Evento detectado en la SQLite_DB, para poder consultar el historial de detecciones.

#### Criterios de Aceptación

1. THE Repositorio_Eventos SHALL crear la SQLite_DB en la ruta configurada si el archivo no existe al iniciar la aplicación.
2. THE Repositorio_Eventos SHALL crear la tabla `eventos` con el esquema: `id INTEGER PRIMARY KEY AUTOINCREMENT`, `timestamp TEXT NOT NULL`, `tipo_maquina TEXT NOT NULL`, `clase_id TEXT NOT NULL`, `confianza REAL NOT NULL`, `numero_fotograma INTEGER NOT NULL`.
3. WHEN el Detector genera un Evento válido, THE Repositorio_Eventos SHALL insertar el Evento en la tabla `eventos` dentro de una transacción atómica.
4. IF una inserción en la tabla `eventos` falla, THEN THE Repositorio_Eventos SHALL registrar el error en consola con los datos del Evento fallido y continuar el procesamiento sin detener la aplicación.
5. THE Repositorio_Eventos SHALL exponer un método `obtener_todos()` que retorne todos los registros de la tabla `eventos` ordenados por `timestamp` ascendente.
6. WHEN se llama al método `obtener_todos()`, THE Repositorio_Eventos SHALL retornar una lista de objetos con los campos definidos en el esquema de la tabla `eventos`.

---

### Requerimiento 5: Registro en consola

**User Story:** Como desarrollador, quiero que el Sistema imprima en consola los eventos detectados en tiempo de procesamiento, para tener visibilidad inmediata sin necesidad de consultar la base de datos.

#### Criterios de Aceptación

1. WHEN el Detector genera un Evento válido, THE Sistema SHALL imprimir en consola una línea con el formato: `[EVENTO] fotograma={numero_fotograma} | timestamp={timestamp} | maquina={tipo_maquina} | clase={clase_id} | confianza={confianza:.2f}`.
2. WHEN ocurre un error durante el procesamiento de un Fotograma, THE Sistema SHALL imprimir en consola una línea con el formato: `[ERROR] fotograma={numero_fotograma} | detalle={descripcion_error}` y continuar con el siguiente fotograma.
3. THE Sistema SHALL imprimir en consola un resumen final al completar el procesamiento con el formato: `[RESUMEN] fotogramas_procesados={n} | eventos_registrados={m} | duracion_segundos={s:.2f}`.

---

### Requerimiento 6: API de consulta de eventos

**User Story:** Como desarrollador, quiero que la API exponga un endpoint para consultar los eventos registrados, para verificar los resultados del procesamiento sin acceder directamente a la base de datos.

#### Criterios de Aceptación

1. THE API SHALL exponer el endpoint `GET /eventos` que retorne todos los Eventos almacenados en la SQLite_DB en formato JSON.
2. WHEN la SQLite_DB no contiene registros, THE API SHALL retornar `GET /eventos` con una lista JSON vacía `[]` y código de estado HTTP `200`.
3. WHEN la SQLite_DB contiene registros, THE API SHALL retornar `GET /eventos` con la lista de Eventos en formato JSON con código de estado HTTP `200`, ordenados por `timestamp` ascendente.
4. THE API SHALL exponer el endpoint `GET /eventos/{id}` que retorne el Evento con el `id` especificado.
5. WHEN el `id` solicitado en `GET /eventos/{id}` no existe en la SQLite_DB, THE API SHALL retornar código de estado HTTP `404` con un mensaje JSON `{"detalle": "Evento no encontrado"}`.
6. IF la SQLite_DB no está disponible al recibir una solicitud, THEN THE API SHALL retornar código de estado HTTP `503` con un mensaje JSON `{"detalle": "Base de datos no disponible"}`.
7. THE API SHALL exponer el endpoint `GET /health` que retorne `{"status": "ok"}` con código de estado HTTP `200` para verificar que el servicio está operativo.

---

### Requerimiento 7: Configuración de la aplicación

**User Story:** Como desarrollador, quiero que los parámetros operativos del Sistema sean configurables sin modificar el código fuente, para facilitar la ejecución en diferentes entornos.

#### Criterios de Aceptación

1. THE Sistema SHALL leer los siguientes parámetros desde variables de entorno o un archivo `.env` en la raíz del proyecto: `VIDEO_PATH`, `MODEL_PATH`, `DB_PATH`, `CONFIDENCE_THRESHOLD`, `API_HOST`, `API_PORT`.
2. WHEN una variable de entorno obligatoria (`VIDEO_PATH`, `MODEL_PATH`, `DB_PATH`) no está definida, THE Sistema SHALL registrar un error descriptivo en consola indicando el nombre de la variable faltante y detener la ejecución.
3. WHEN `CONFIDENCE_THRESHOLD` no está definido, THE Sistema SHALL utilizar el valor predeterminado `0.5`.
4. WHEN `API_HOST` o `API_PORT` no están definidos, THE Sistema SHALL utilizar los valores predeterminados `127.0.0.1` y `8000` respectivamente.

---

### Requerimiento 8: Estructura y organización del proyecto

**User Story:** Como desarrollador, quiero que el proyecto siga una estructura de directorios modular y escalable, para que cada componente tenga una responsabilidad clara y el código sea fácil de mantener y extender.

#### Criterios de Aceptación

1. THE Sistema SHALL contener los siguientes directorios en la raíz del proyecto: `data/raw/`, `data/processed/`, `models/`, `src/ai/`, `src/core/`, `src/database/`, `src/api/routers/`, `reports/`.
2. THE `src/config.py` SHALL centralizar toda la configuración de la aplicación, incluyendo variables de entorno, rutas de archivos y parámetros del modelo YOLO.
3. THE `src/ai/detector.py` SHALL contener exclusivamente la clase `Detector` con la lógica de inferencia YOLO y el procesamiento de fotogramas.
4. THE `src/database/models.py` SHALL definir los modelos ORM de la tabla `eventos` y cualquier otra entidad persistida en la SQLite_DB.
5. THE `src/database/database.py` SHALL gestionar la conexión a la SQLite_DB y el ciclo de vida de las sesiones de base de datos.
6. THE `src/api/main.py` SHALL ser el punto de entrada de la aplicación FastAPI, responsable de registrar los routers y configurar la instancia de la aplicación.
7. THE `src/api/routers/` SHALL contener un módulo Python separado por cada dominio funcional expuesto por la API.
8. THE `src/api/schemas.py` SHALL definir todos los modelos Pydantic utilizados para la validación de datos de entrada y la serialización de respuestas de la API.
9. THE `requirements.txt` SHALL listar todas las dependencias del proyecto con versiones fijadas (pinned) en formato `paquete==version`.
10. THE `README.md` SHALL existir en la raíz del proyecto e incluir al menos las instrucciones mínimas de instalación del entorno y ejecución de la aplicación.
