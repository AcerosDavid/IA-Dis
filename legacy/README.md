# Código archivado (legacy)

Este directorio conserva componentes de la etapa anterior basada en servidor web.
**No forma parte de la aplicación de escritorio actual** y no está pensado para
ejecutarse tal cual: sus imports apuntan a módulos que ya no existen en `src/`.

Contenido:

- `api/` — backend FastAPI, routers, schemas y utilidades de video.
- `auth/` — autenticación JWT / bcrypt.
- `cli/` — pipeline CLI anterior (`main.py`) y cámara (`camera.py`).
- `infra/` — migraciones Alembic, Docker y variables de despliegue.
- `tests/` — tests de la API eliminada.
- `docs/` — documentación de despliegue y especificaciones antiguas.

Si en el futuro se necesita recuperar alguna funcionalidad, se puede migrar
selectivamente a `src/` adaptando los imports.
