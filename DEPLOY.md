# Mining Dispatch API — Guía de Deploy en Servidor GPU

## Requisitos del servidor

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU NVIDIA | GTX 1060 6GB | RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Driver NVIDIA | ≥ 525 | último |
| Docker Engine | 24+ | último |
| NVIDIA Container Toolkit | requerido | requerido |

---

## 1. Instalar NVIDIA Container Toolkit (una vez por servidor)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verificar
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. Copiar el proyecto al servidor

```bash
# Opción A — clonar desde git
git clone <repo-url> IA-Dis
cd IA-Dis

# Opción B — rsync desde tu máquina local (Windows → Linux)
rsync -avz --exclude='.venv' --exclude='data/' --exclude='runs/' \
  /mnt/c/Users/David\ Aceros/Documents/Proyectos/Estudio/IA/IA-Dis/ \
  usuario@servidor-ip:/opt/mining-dispatch/
```

---

## 3. Copiar el modelo entrenado

```bash
# Desde tu máquina local
scp "C:\Users\David Aceros\Documents\Proyectos\Estudio\IA\IA-Dis\models\best.pt" \
    usuario@servidor-ip:/opt/mining-dispatch/models/best.pt
```

---

## 4. Configurar variables de entorno

```bash
cd /opt/mining-dispatch
cp .env.docker .env.docker.local
nano .env.docker.local   # ajustar CONFIDENCE_THRESHOLD si necesario
```

---

## 5. Build y arranque

```bash
cd /opt/mining-dispatch

# Build de la imagen (solo la primera vez o tras cambios en código)
docker compose build

# Arrancar con GPU
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f api

# Verificar que detecta la GPU
docker compose exec api python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 6. Verificar que funciona

```bash
# Health check
curl http://servidor-ip:8000/eventos/health

# Ver docs de la API
# Abrir en navegador: http://servidor-ip:8000/docs
```

---

## 7. Conectar el frontend al servidor

Editar `Mining_dispatchIA/.env`:
```
VITE_API_URL=http://servidor-ip:8000
```

Reconstruir el frontend:
```powershell
cd "C:\Users\David Aceros\Documents\Proyectos\Estudio\IA\Mining_dispatchIA"
npm run build
```

---

## Comandos útiles

```bash
# Ver uso de GPU en tiempo real
watch -n1 nvidia-smi

# Reiniciar el contenedor
docker compose restart api

# Ver logs de los últimos 100 líneas
docker compose logs --tail=100 api

# Parar todo
docker compose down

# Parar y borrar datos (¡cuidado!)
docker compose down -v

# Actualizar imagen tras cambios en código
docker compose build --no-cache && docker compose up -d

# Entrar al contenedor para debug
docker compose exec api bash

# Backup de la base de datos
docker compose exec api cp /app/data/eventos.db /tmp/backup.db
docker cp mining-dispatch-api:/tmp/backup.db ./backup_$(date +%Y%m%d).db
```

---

## Ajuste de rendimiento por GPU

Editar `src/api/routers/video.py`:

| GPU | TRACK_IMGSZ | TRACK_SKIP | FPS esperado |
|---|---|---|---|
| GTX 1060 6GB | 480 | 1 | ~20 fps |
| RTX 3060 12GB | 640 | 1 | ~35 fps |
| RTX 3070/3080 | 640 | 1 | ~50+ fps |
| RTX 4080/4090 | 640 | 1 | ~80+ fps |
| A100 80GB | 1280 | 1 | ~120+ fps |

---

## Troubleshooting

**Error: "could not select device driver nvidia"**
→ NVIDIA Container Toolkit no está instalado o el daemon Docker no se reinició.

**Error: "CUDA out of memory"**
→ Bajar `TRACK_IMGSZ` a 320 en `video.py`.

**El contenedor arranca pero YOLO usa CPU**
→ Verificar con `docker compose exec api python -c "import torch; print(torch.cuda.is_available())"`.
→ Si da `False`, el runtime NVIDIA no está configurado.

**Timeout al subir videos grandes**
→ Agregar nginx como proxy con `client_max_body_size 600m` delante del contenedor.
