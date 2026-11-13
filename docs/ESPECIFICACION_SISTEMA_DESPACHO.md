# Especificación funcional y técnica

## Sistema inteligente de análisis de video para despacho de flotas mineras

**Versión:** 1.0  
**Estado:** Documento base para completar el backend y construir el frontend  
**Objetivo:** Definir los módulos, funcionalidades, contratos de API, estados, seguridad y criterios de aceptación necesarios para una plataforma que procese videos extensos en segundo plano y genere reportes operacionales.

---

## 1. Resumen del sistema

La plataforma permitirá cargar uno o varios videos de operaciones mineras, configurar qué análisis ejecutar, procesarlos de forma asíncrona, consultar el progreso, revisar los eventos detectados y generar reportes descargables.

El sistema debe conservar la relación entre cada video, cada ejecución de análisis, el modelo y su versión, los eventos encontrados y los reportes generados.

### Objetivos principales

- Procesar videos de varias horas sin mantener abierta una petición HTTP durante todo el análisis.
- Admitir múltiples trabajos y controlar la concurrencia según la capacidad disponible.
- Mostrar progreso real de carga y procesamiento.
- Permitir consultar, filtrar y recuperar videos, ejecuciones, eventos y reportes históricos.
- Facilitar la validación humana de los hallazgos.
- Mantener trazabilidad, seguridad y observabilidad.
- Permitir que el usuario abandone la pantalla y regrese después sin perder el estado del trabajo.

### Principios de diseño

1. El backend es la fuente de verdad de estados, porcentajes, resultados y errores.
2. El frontend no simula progreso ni inventa resultados.
3. El análisis se ejecuta en workers o procesos independientes de la petición web.
4. Cada ejecución debe poder identificarse, consultarse y auditarse.
5. Solo deben ofrecerse opciones de análisis que estén soportadas por los modelos y servicios existentes.
6. Los resultados automáticos deben distinguirse de los eventos confirmados por una persona.

---

## 2. Roles y permisos

Los roles sugeridos son una base; se pueden adaptar a los usuarios actuales.

| Rol | Permisos principales |
|---|---|
| Administrador | Gestionar usuarios, permisos, configuración global y modelos |
| Supervisor | Consultar operaciones, revisar eventos y generar reportes |
| Analista | Cargar videos, ejecutar análisis y revisar resultados |
| Consulta | Ver videos, eventos y reportes autorizados, sin modificar trabajos |

El backend debe validar los permisos en cada endpoint. Ocultar botones en el frontend no sustituye la autorización del servidor.

---

## 3. Módulos del frontend

### 3.1 Dashboard operativo

Debe mostrar un resumen del estado del sistema y de la actividad reciente.

**Indicadores sugeridos:**

- Videos cargados en el periodo seleccionado.
- Trabajos pendientes, procesando, completados y fallidos.
- Horas de video analizadas.
- Duración promedio de procesamiento y relación entre duración del video y tiempo de análisis, cuando esos datos estén disponibles.
- Eventos detectados por categoría y periodo.
- Eventos pendientes de validación.
- Trabajos recientes y reportes recientes.
- Estado de los servicios de procesamiento, si el backend expone esa información.

**Filtros:** rango de fechas, mina/proyecto, turno, equipo/cámara y categoría de análisis.

Los indicadores deben calcularse a partir de datos reales del backend.

### 3.2 Centro de carga y análisis de videos

Debe permitir:

- Cargar uno o varios archivos.
- Arrastrar y soltar archivos.
- Mostrar nombre, tamaño, duración y estado de cada archivo, cuando se conozcan.
- Validar formatos, tamaño máximo y requisitos definidos por el backend.
- Reanudar cargas interrumpidas si la infraestructura lo soporta.
- Asociar metadatos: mina/proyecto, equipo o cámara, fecha, turno, ubicación y notas.
- Seleccionar un perfil de análisis o configurar opciones avanzadas.
- Elegir únicamente modelos y capacidades compatibles.
- Revisar la configuración antes de enviar el trabajo.
- Mostrar por separado el progreso de carga y el progreso del análisis.
- Evitar envíos duplicados accidentales.
- Informar errores de validación con mensajes accionables.

**Opciones de análisis posibles, solo si están implementadas en el backend:**

- Clasificación de actividad operacional.
- Medición de tiempos de ciclo, espera o actividad.
- Detección y clasificación de eventos.
- Identificación o seguimiento de vehículos.
- Análisis de condiciones de seguridad.
- Detección de presencia de personas o ingreso a zonas.
- Generación de resumen ejecutivo.
- Otras tareas específicas del dominio minero.

La interfaz debe construir las opciones desde las capacidades expuestas por el backend cuando sea posible, en lugar de codificar capacidades inexistentes.

### 3.3 Cola de procesamiento

Debe listar los trabajos con estados actualizados.

**Acciones posibles según permisos y estado:**

- Abrir detalle.
- Cancelar un trabajo si todavía es cancelable.
- Reintentar un trabajo fallido.
- Cambiar prioridad si el backend lo permite.
- Consultar mensajes de error.
- Filtrar por estado, fecha, usuario, modelo y proyecto.

Cada trabajo debe mostrar como mínimo: identificador, video, estado, fecha de creación, inicio y finalización si existen, progreso, modelo/configuración y usuario que lo inició.

### 3.4 Historial de videos y análisis

Debe permitir:

- Buscar por nombre o identificador.
- Filtrar por mina, equipo/cámara, turno, fecha, estado y categoría.
- Abrir el detalle del video.
- Ver todas las ejecuciones asociadas al mismo video.
- Consultar los modelos y parámetros usados en cada ejecución.
- Abrir resultados y reportes históricos.
- Identificar ejecuciones fallidas o incompletas.
- Reprocesar un video con una nueva configuración, creando una nueva ejecución sin borrar la anterior.

Se debe diferenciar entre el **archivo de video** y las **ejecuciones de análisis** realizadas sobre ese archivo.

### 3.5 Detalle del video y reproductor

Debe incluir:

- Reproductor de video con controles estándar.
- Navegación a un segundo específico.
- Línea de tiempo con marcadores de eventos, si existen.
- Lista de eventos asociados al video.
- Filtros por categoría, severidad, confianza y estado de revisión.
- Acceso a capturas o clips de evidencia si el backend los genera.
- Metadatos del video y del origen.
- Historial de análisis realizados.
- Estado de disponibilidad del archivo y permisos de acceso.

El reproductor debe buscar el instante correspondiente al evento al seleccionarlo. No se debe asumir que el backend genera clips si solo guarda marcas de tiempo.

### 3.6 Explorador y revisión de eventos

Cada evento debe poder mostrar, cuando los datos existan:

- Categoría y descripción.
- Tiempo de inicio y fin, o marca de tiempo.
- Confianza del modelo.
- Severidad o prioridad, si está definida.
- Evidencia asociada.
- Modelo y versión que lo detectaron.
- Estado de revisión.
- Comentarios y usuario revisor.

**Acciones de revisión:**

- Confirmar.
- Rechazar como falso positivo.
- Corregir categoría o datos, si se permite.
- Añadir comentario.
- Marcar para seguimiento.

Los cambios deben quedar auditados. La confianza del modelo no debe presentarse como una probabilidad calibrada si el modelo no la produce con ese significado.

### 3.7 Reportes

Debe permitir generar, consultar y descargar reportes.

**Contenido sugerido:**

- Identificación del video y metadatos operacionales.
- Resumen ejecutivo.
- Configuración, modelo y versión usados.
- Duración del video y tiempos del procesamiento.
- Estadísticas y gráficos.
- Lista de eventos con marcas de tiempo.
- Evidencias disponibles.
- Estado de revisión humana.
- Limitaciones o advertencias del análisis.
- Fecha de generación y autor.

**Formatos posibles:** PDF, XLSX y CSV, según las necesidades del usuario.

El reporte debe generarse a partir de resultados persistidos. Si la generación es costosa, también debe ser un trabajo asíncrono.

### 3.8 Administración de modelos

Si aplica a la infraestructura actual, debe mostrar:

- Nombre, descripción y versión.
- Capacidades soportadas.
- Estado de disponibilidad.
- Parámetros configurables.
- Fecha de publicación o actualización.
- Compatibilidad con tipos de video.
- Métricas de rendimiento y calidad disponibles.

La interfaz no debe permitir cambiar pesos, desplegar versiones o modificar parámetros que el backend no permita controlar de forma segura.

### 3.9 Gestión de flota y contexto operacional

Si el negocio necesita segmentar resultados por operación, incluir catálogos para:

- Minas o proyectos.
- Equipos y vehículos.
- Cámaras y fuentes de video.
- Turnos.
- Zonas operacionales.
- Tipos de actividad.

Estos catálogos pueden integrarse con sistemas existentes o administrarse localmente. Evitar duplicar información maestra si ya existe otro sistema responsable de ella.

### 3.10 Usuarios, auditoría y configuración

Según los requisitos del proyecto:

- Usuarios y roles.
- Permisos por proyecto o mina.
- Historial de acciones relevantes.
- Parámetros de carga y procesamiento.
- Políticas de retención de archivos.
- Configuración de notificaciones.
- Estado de servicios y dependencias.

---

## 4. Requisitos del backend

### 4.1 API y validación

El backend debe proporcionar endpoints claros y documentados, preferiblemente con OpenAPI si se usa una tecnología compatible.

Debe validar:

- Autenticación y autorización.
- Tipo y tamaño de archivo.
- Metadatos requeridos.
- Compatibilidad entre modelo, opciones y video.
- Límites de concurrencia y cuotas.
- Transiciones válidas de estado.
- Entradas de filtros, paginación y ordenamiento.
- Permisos sobre videos, resultados, evidencias y reportes.

Los errores deben devolver un formato consistente con código, mensaje legible y, cuando aplique, detalles de validación y un identificador de correlación.

### 4.2 Carga y almacenamiento de videos

El backend debe definir:

- Formatos de video aceptados.
- Tamaño máximo y duración máxima, si existen límites.
- Estrategia de carga directa, carga multipartes o carga reanudable.
- Almacenamiento de archivos originales.
- Validación de integridad, por ejemplo mediante checksum.
- Detección de cargas incompletas o duplicadas.
- Política de retención y eliminación.
- Acceso controlado a archivos y evidencias.

Para archivos grandes, es preferible evitar que el servidor de API actúe como intermediario de todo el flujo si se puede utilizar almacenamiento de objetos con URLs temporales y controladas.

### 4.3 Procesamiento asíncrono

El análisis debe ejecutarse fuera del ciclo de vida de la petición HTTP, mediante una cola y workers o la infraestructura equivalente ya utilizada por el proyecto.

Capacidades requeridas:

- Crear un trabajo y devolver su identificador rápidamente.
- Persistir el estado del trabajo.
- Procesar en segundo plano.
- Limitar la concurrencia según CPU, GPU, memoria y capacidad de almacenamiento.
- Registrar progreso real cuando el pipeline pueda calcularlo.
- Manejar fallos y reintentos con límites.
- Permitir cancelación cuando sea técnicamente segura.
- Evitar procesar dos veces el mismo trabajo por reintentos de red.
- Recuperar trabajos tras reinicios de servicios.
- Registrar duración, recursos y errores.
- Separar errores transitorios de errores permanentes.

No se debe prometer una velocidad de procesamiento determinada hasta medirla con videos reales y el hardware objetivo.

### 4.4 Progreso y comunicación con el frontend

El backend debe ofrecer un mecanismo para consultar el estado:

- Endpoint de estado consultable periódicamente; y/o
- Server-Sent Events (SSE); y/o
- WebSocket.

El estado debería incluir porcentaje solo cuando se pueda calcular con una base real. Si no es posible estimarlo, se debe mostrar una etapa como "Preparando", "Analizando" o "Generando reporte", sin inventar porcentajes ni tiempos restantes.

La comunicación en tiempo real es una mejora de experiencia, pero no debe ser la única forma de recuperar el estado. Al recargar la página, el frontend debe poder consultar el trabajo desde la API.

### 4.5 Persistencia de resultados

Guardar de forma duradera:

- Metadatos del video.
- Estado y configuración de cada ejecución.
- Modelo y versión utilizados.
- Eventos detectados.
- Evidencias y sus referencias.
- Resultados agregados y métricas.
- Reportes generados.
- Errores relevantes.
- Revisiones y comentarios humanos.
- Auditoría de acciones importantes.

No almacenar grandes archivos binarios en tablas relacionales salvo que exista una razón técnica explícita. Es preferible guardar archivos en almacenamiento apropiado y conservar sus referencias y metadatos en la base de datos.

### 4.6 Generación de reportes

El servicio de reportes debe:

- Recibir el identificador de la ejecución.
- Validar permisos y disponibilidad de resultados.
- Generar el formato solicitado.
- Guardar el archivo generado.
- Registrar fecha, formato, versión de plantilla y estado.
- Permitir consultar progreso si la generación es asíncrona.
- Entregar una descarga autorizada o un enlace temporal seguro.
- Gestionar errores y reintentos.

### 4.7 Observabilidad y administración

Registrar métricas y logs para conocer:

- Trabajos por estado.
- Tiempo de espera en cola.
- Duración de procesamiento por modelo.
- Duración del video frente al tiempo de procesamiento.
- Tasa de errores y reintentos.
- Consumo de CPU, GPU, memoria y disco cuando sea posible.
- Capacidad de la cola.
- Fallos de almacenamiento y generación de reportes.

No exponer datos sensibles, tokens, URLs firmadas completas ni contenido confidencial en logs.

---

## 5. Modelo de datos conceptual

Este modelo es una propuesta inicial que debe adaptarse a la base de datos y entidades existentes.

### 5.1 Video

Campos sugeridos:

- `id`
- `original_filename`
- `storage_key` o referencia de almacenamiento
- `content_type`
- `file_size_bytes`
- `duration_seconds`
- `checksum`
- `uploaded_by`
- `uploaded_at`
- `project_id`
- `equipment_id` o `camera_id`
- `shift_id`
- `recorded_at`
- `metadata`
- `retention_status`

### 5.2 AnalysisJob

Representa una ejecución concreta, no el archivo original.

- `id`
- `video_id`
- `status`
- `stage`
- `progress_percent`, nullable
- `analysis_options`
- `model_id`
- `model_version`
- `created_by`
- `created_at`
- `started_at`
- `finished_at`
- `error_code`
- `error_message`
- `retry_count`
- `queue_priority`, si aplica
- `metrics`

### 5.3 DetectedEvent

- `id`
- `analysis_job_id`
- `category`
- `description`
- `start_time_seconds`
- `end_time_seconds`, nullable
- `confidence`, nullable
- `severity`, nullable
- `evidence_reference`, nullable
- `review_status`
- `reviewed_by`, nullable
- `reviewed_at`, nullable
- `review_comment`, nullable
- `created_at`

### 5.4 Report

- `id`
- `analysis_job_id`
- `format`
- `status`
- `storage_key`, nullable
- `created_by`
- `created_at`
- `finished_at`
- `error_message`, nullable
- `template_version`

### 5.5 ModelRegistry

- `id`
- `name`
- `version`
- `description`
- `capabilities`
- `configuration_schema`
- `status`
- `created_at`
- `updated_at`

### 5.6 Auditoría

Registrar actor, acción, recurso, fecha, resultado y contexto no sensible. No permitir que los usuarios alteren o borren la auditoría mediante operaciones ordinarias.

---

## 6. Estados y transiciones

### 6.1 Estados de análisis

Estados sugeridos:

- `uploaded`: archivo cargado y registrado.
- `queued`: esperando recursos para procesar.
- `processing`: análisis en ejecución.
- `generating_report`: generando un reporte, si forma parte del mismo flujo.
- `completed`: ejecución finalizada correctamente.
- `failed`: ejecución fallida.
- `cancel_requested`: cancelación solicitada, si el proceso requiere tiempo para detenerse.
- `cancelled`: ejecución cancelada.

Definir las transiciones válidas en el backend. Por ejemplo, un trabajo completado no debería volver a `processing`; un reprocesamiento debe crear una ejecución nueva.

Si los reportes se gestionan como recursos independientes, su estado debe mantenerse separado del estado del análisis.

### 6.2 Carga de archivos

Los estados de carga son independientes de los estados de análisis. Ejemplo:

- `pending`
- `uploading`
- `uploaded`
- `upload_failed`
- `validating`

La interfaz no debe mostrar "procesando" cuando el archivo todavía se está cargando.

---

## 7. Contrato de API sugerido

Los nombres son ejemplos; deben adaptarse a las rutas existentes. No es necesario crear endpoints duplicados si el backend ya ofrece funciones equivalentes.

| Método | Ruta sugerida | Propósito |
|---|---|---|
| `POST` | `/api/videos` | Registrar video o iniciar flujo de carga |
| `POST` | `/api/uploads` | Iniciar una carga multipartes/reanudable, si aplica |
| `GET` | `/api/uploads/{id}` | Consultar estado de carga |
| `POST` | `/api/videos/{id}/analysis-jobs` | Crear una ejecución de análisis |
| `GET` | `/api/analysis-jobs` | Listar y filtrar trabajos |
| `GET` | `/api/analysis-jobs/{id}` | Consultar detalle y estado |
| `GET` | `/api/analysis-jobs/{id}/events` | Consultar eventos detectados |
| `POST` | `/api/analysis-jobs/{id}/cancel` | Solicitar cancelación |
| `POST` | `/api/analysis-jobs/{id}/retry` | Reintentar si está permitido |
| `GET` | `/api/videos` | Buscar y listar videos |
| `GET` | `/api/videos/{id}` | Consultar metadatos del video |
| `GET` | `/api/videos/{id}/analysis-jobs` | Consultar ejecuciones del video |
| `PATCH` | `/api/events/{id}/review` | Guardar revisión humana |
| `POST` | `/api/analysis-jobs/{id}/reports` | Solicitar generación de reporte |
| `GET` | `/api/reports` | Listar reportes |
| `GET` | `/api/reports/{id}` | Consultar estado y metadatos |
| `GET` | `/api/reports/{id}/download` | Descargar reporte con autorización |
| `GET` | `/api/models` | Consultar modelos y capacidades |
| `GET` | `/api/dashboard/summary` | Obtener indicadores agregados |

### Ejemplo de creación de un trabajo

Petición ilustrativa:

```json
{
  "video_id": "video_123",
  "model_id": "operational-analysis",
  "model_version": "1.2.0",
  "options": {
    "detect_events": true,
    "calculate_cycle_times": true
  },
  "metadata": {
    "project_id": "mine_01",
    "shift_id": "night"
  }
}
```

Respuesta ilustrativa:

```json
{
  "job_id": "job_456",
  "status": "queued",
  "created_at": "2026-10-20T20:00:00Z",
  "status_url": "/api/analysis-jobs/job_456"
}
```

La respuesta real debe usar los identificadores, formatos de fecha y campos de la implementación existente.

### Ejemplo de estado

```json
{
  "id": "job_456",
  "status": "processing",
  "stage": "detecting_events",
  "progress_percent": 42,
  "started_at": "2026-10-20T20:01:00Z",
  "finished_at": null,
  "error": null
}
```

Si no existe un porcentaje real, devolver `null` y mostrar la etapa del trabajo.

### Paginación y filtros

Los endpoints de listas deben admitir, según aplique:

- `page` y `page_size`, o cursor.
- `status`
- `from` y `to`
- `project_id`
- `equipment_id` o `camera_id`
- `model_id`
- `created_by`
- `sort_by` y `sort_order`

Definir límites máximos de página y devolver metadatos de paginación consistentes.

---

## 8. Requisitos del frontend

### 8.1 Arquitectura sugerida

Una estructura posible para React y TypeScript:

```text
src/
  app/
    router/
    providers/
    layouts/
  components/
    ui/
    data-display/
    forms/
    video/
    charts/
  features/
    dashboard/
    video-upload/
    analysis-jobs/
    video-history/
    events/
    reports/
    models/
    fleet/
    users/
    settings/
  services/
    api-client.ts
    videos.api.ts
    analysis-jobs.api.ts
    events.api.ts
    reports.api.ts
    models.api.ts
  hooks/
  types/
  utils/
```

La estructura es orientativa, no obligatoria. Si el frontend ya tiene una arquitectura estable, conviene extenderla en vez de reescribirla sin necesidad.

### 8.2 Páginas y rutas sugeridas

- `/dashboard`
- `/videos/upload`
- `/jobs`
- `/jobs/:jobId`
- `/videos`
- `/videos/:videoId`
- `/videos/:videoId/analyses`
- `/events`
- `/reports`
- `/reports/:reportId`
- `/models`
- `/fleet`
- `/users`
- `/settings`

Proteger rutas según el rol, además de validar los permisos en el backend.

### 8.3 Estados de interfaz obligatorios

Cada página debe manejar:

- Cargando.
- Vacío, sin resultados.
- Datos cargados.
- Error de red o del servidor.
- Acceso denegado.
- Recurso inexistente.
- Trabajo en progreso.
- Trabajo fallido con explicación y acción posible.
- Datos parcialmente disponibles.

No mostrar pantallas en blanco ni dejar botones habilitados durante acciones incompatibles.

### 8.4 Experiencia visual

Estilo recomendado:

- Interfaz industrial, sobria y de alta legibilidad.
- Navegación lateral persistente.
- Dashboard con tarjetas de indicadores y gráficos moderados.
- Tablas con filtros, búsqueda, ordenamiento y paginación.
- Estados visibles mediante texto e iconos, no solo colores.
- Diseño adaptable a escritorio y tablet.
- Accesibilidad por teclado y contraste adecuado.
- Fechas, horas y zonas horarias presentadas consistentemente.

### 8.5 Integración con API

- Centralizar la configuración de base URL, autenticación y manejo de errores.
- Tipar las respuestas y solicitudes.
- Cancelar solicitudes de lectura obsoletas cuando corresponda.
- Invalidar o actualizar caché al cambiar estados.
- Reintentar automáticamente solo solicitudes seguras y fallos transitorios.
- Evitar reintentar acciones de creación sin una clave de idempotencia o estrategia equivalente.
- Mantener el estado de los trabajos en el servidor, no solo en memoria del navegador.

---

## 9. Seguridad, privacidad y trazabilidad

Requisitos mínimos:

- Autenticación segura y expiración/renovación controlada de sesión.
- Autorización por rol y por recurso.
- Validación del contenido real del archivo, no solo su extensión.
- Límites de tamaño y protección contra cargas abusivas.
- URLs temporales para descargas cuando corresponda.
- Protección frente a accesos directos no autorizados a videos, evidencias y reportes.
- CORS configurado para los orígenes autorizados.
- Secretos y credenciales únicamente en el servidor.
- No incluir tokens en logs, URLs permanentes o mensajes de error.
- Política de retención y eliminación de videos.
- Registro de quién cargó, analizó, revisó, exportó o eliminó recursos.
- Protección de datos personales que puedan aparecer en las grabaciones.
- Copias de seguridad y recuperación para metadatos y resultados críticos.

Si se detectan eventos relacionados con seguridad operacional, la interfaz debe dejar claro que son hallazgos automáticos pendientes de validación cuando corresponda. No debe presentar una inferencia del modelo como un incidente confirmado sin revisión definida por el negocio.

---

## 10. Pruebas necesarias

### 10.1 Backend

- Carga válida, inválida, incompleta y reanudada.
- Creación de trabajos con opciones compatibles e incompatibles.
- Persistencia de estados y recuperación tras reinicios.
- Reintentos y fallos de workers.
- Cancelación en distintas etapas.
- Idempotencia ante solicitudes duplicadas.
- Control de acceso por usuario y rol.
- Consultas con filtros y paginación.
- Generación y descarga de reportes.
- Videos largos, archivos grandes y formatos límite.
- Carga simultánea y límites de recursos.

### 10.2 Frontend

- Formularios y validaciones.
- Carga múltiple y mensajes de error.
- Actualización de progreso y reconexión.
- Recarga de la página mientras un trabajo está activo.
- Filtros, paginación y búsqueda.
- Estados vacíos y errores del servidor.
- Navegación desde un evento a su segundo exacto del video.
- Descarga de reportes.
- Permisos y acciones no disponibles.
- Uso con teclado y diseño adaptable.

### 10.3 Pruebas de rendimiento

Medir con muestras reales:

- Tiempo de carga de archivos.
- Duración del análisis frente a la duración del video.
- Tiempo de espera en cola.
- Consumo máximo de CPU, GPU y memoria.
- Rendimiento con varios videos simultáneos.
- Tiempo de consulta de historiales grandes.
- Tiempo de generación de reportes.

Definir objetivos de rendimiento después de obtener una línea base con el hardware, los modelos y los tamaños de video reales.

---

## 11. Criterios de aceptación

La primera versión se considera funcional cuando:

1. Un usuario autorizado puede cargar un video válido y consultar el estado de la carga.
2. Puede iniciar un análisis con opciones soportadas.
3. El backend crea un trabajo persistente y devuelve un identificador.
4. El análisis continúa aunque el usuario cierre la pestaña.
5. El frontend muestra estados reales y recupera el trabajo después de recargar.
6. Los trabajos fallidos muestran información útil y permiten reintentar cuando sea válido.
7. El usuario puede consultar el historial y abrir el detalle de una ejecución.
8. Los eventos persistidos se muestran asociados al video y a la ejecución correcta.
9. Se puede generar y descargar un reporte autorizado.
10. Se conserva la relación entre video, ejecución, modelo, eventos y reporte.
11. Las acciones de revisión humana quedan registradas.
12. Los usuarios no pueden acceder a recursos para los que no tienen permiso.
13. La interfaz diferencia carga, procesamiento y generación del reporte.
14. Los datos de dashboard y las estadísticas corresponden a información real.
15. Los errores y las limitaciones del sistema se comunican de forma comprensible.

---

## 12. Plan de implementación

### Fase 1 — Inventario y contratos

- Revisar el backend y la documentación de API existente.
- Identificar qué funcionalidades ya están implementadas.
- Documentar formatos de video, límites, modelos y opciones.
- Definir entidades y estados reales.
- Acordar los contratos de API que faltan.

**Entregable:** matriz de funcionalidades existentes, faltantes y cambios necesarios.

### Fase 2 — Flujo principal

- Carga de videos.
- Creación de trabajos asíncronos.
- Cola e indicador de estado.
- Historial básico.
- Detalle de ejecución y errores.

**Entregable:** flujo completo desde la carga hasta el resultado persistido.

### Fase 3 — Resultados y reportes

- Explorador de eventos.
- Reproductor con navegación por marcas de tiempo.
- Revisión humana.
- Generación y descarga de reportes.
- Historial de ejecuciones por video.

**Entregable:** ciclo completo de análisis y consulta de resultados.

### Fase 4 — Dashboard y administración

- Indicadores agregados.
- Filtros avanzados.
- Gestión de modelos y catálogos.
- Usuarios y permisos.
- Auditoría y notificaciones.

**Entregable:** plataforma de operación y administración.

### Fase 5 — Endurecimiento y despliegue

- Pruebas de carga y recuperación.
- Seguridad y permisos.
- Monitoreo y alertas.
- Copias de seguridad.
- Documentación de despliegue y operación.

**Entregable:** versión preparada para un entorno operativo controlado.

---

## 13. Matriz para revisar el backend actual

Completar esta tabla antes de desarrollar funcionalidades nuevas.

| Capacidad | Estado actual | Evidencia o endpoint | Pendiente |
|---|---|---|---|
| Autenticación y autorización | Por verificar | | |
| Carga de videos grandes | Por verificar | | |
| Almacenamiento persistente | Por verificar | | |
| Creación de trabajos asíncronos | Por verificar | | |
| Cola y workers | Por verificar | | |
| Consulta de progreso | Por verificar | | |
| Cancelación y reintentos | Por verificar | | |
| Registro de ejecuciones históricas | Por verificar | | |
| Persistencia de eventos | Por verificar | | |
| Evidencias y marcas de tiempo | Por verificar | | |
| Generación de reportes | Por verificar | | |
| Descarga autorizada | Por verificar | | |
| Catálogo de modelos y opciones | Por verificar | | |
| Métricas y logs | Por verificar | | |
| Auditoría y revisión humana | Por verificar | | |

---

## 14. Decisiones que deben confirmarse

Antes de cerrar la arquitectura técnica, confirmar:

1. Lenguaje, framework y estructura actuales del backend.
2. Framework y estado actual del frontend.
3. Base de datos y sistema de almacenamiento de videos.
4. Tecnología de cola y ejecución de workers, si existe.
5. Modelos disponibles y tareas que realmente soportan.
6. Si el análisis devuelve progreso numérico o solo etapas.
7. Cómo se representan los eventos y evidencias.
8. Formatos de reporte requeridos.
9. Tamaño máximo, formatos y cantidad habitual de videos simultáneos.
10. Infraestructura de despliegue, hardware disponible y requisitos de seguridad.
11. Roles de usuario y reglas de acceso.
12. Política de conservación y eliminación de videos.

---

## 15. Recomendación final

No es necesario reescribir el backend para implementar esta plataforma. Primero hay que mapear lo que ya existe y cerrar los huecos críticos: trabajos asíncronos persistentes, consulta de estado, historial, resultados y reportes.

La prioridad es lograr un flujo completo y confiable:

**Cargar video → configurar análisis → crear trabajo → procesar en segundo plano → consultar resultados → revisar eventos → generar reporte → recuperar todo desde el historial.**

Una vez estable, se agregan analítica operacional, comparación de modelos, administración de flota y automatizaciones.

Este documento es una especificación inicial. Los nombres de rutas, entidades, estados y campos son propuestas y deben ajustarse al backend real antes de implementarlos.
