"""
lstm_estados.py — Red LSTM bidireccional para clasificar estados operacionales.

Problema central:
  Una foto estática NO distingue "excavadora escarbando" de "excavadora quieta".
  Analizamos la SECUENCIA temporal de vectores de los últimos N frames del mismo
  track_id. La LSTM aprende los patrones de movimiento en el tiempo.

Arquitectura:
  Entrada : (batch, seq_len, feature_dim)
  LSTM    : 2 capas bidireccionales, dropout=0.3
  Cabeza  : LayerNorm → Dropout → Linear → GELU → Linear → logits

Estados por tipo:
  excavadora / pala : quieta | escarbando_cargando | descargando
  camion            : vacio_andando | lleno_andando | quieto_en_cargue_descargue

Uso rápido:
    modelo = LSTMEstados.crear("excavadora")
    logits = modelo(tensor_secuencia)        # (B, n_clases)
    estado, conf = modelo.predecir(tensor)   # inferencia de 1 ejemplo
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── Taxonomía de estados ──────────────────────────────────────────────────────

ESTADOS_POR_TIPO: dict[str, list[str]] = {
    # El tracker detecta si se mueve o está quieto.
    # El clasificador LSTM solo necesita el estado de carga/operación.
    "excavadora": ["escarbando_cargando", "descargando"],
    "pala":       ["escarbando_cargando", "descargando"],
    "camion":     ["vacio", "cargado"],
}

# Aliases YOLO → tipo canónico
_ALIAS: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "truck":        "camion",
    "shovel":       "pala",
}

# Hiperparámetros por defecto
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3


# ── Red neuronal ──────────────────────────────────────────────────────────────

class LSTMEstados(nn.Module):
    """
    LSTM bidireccional para clasificar secuencias temporales de características.

    Args:
        feature_dim:   Dimensión del vector de entrada por frame.
        clases:        Lista de nombres de estados posibles.
        hidden_size:   Neuronas internas de la LSTM.
        num_layers:    Capas apiladas.
        dropout:       Tasa de dropout entre capas.
    """

    def __init__(
        self,
        feature_dim:  int,
        clases:       list[str],
        hidden_size:  int   = HIDDEN_SIZE,
        num_layers:   int   = NUM_LAYERS,
        dropout:      float = DROPOUT,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.clases      = clases
        self.n_clases    = len(clases)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Normalizar entrada frame a frame
        self.norm_entrada = nn.LayerNorm(feature_dim)

        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        lstm_out_dim = hidden_size * 2   # bidireccional → doble

        # Cabeza de clasificación
        self.cabeza = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, self.n_clases),
        )
        self._init_pesos()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            logits: (batch, n_clases)
        """
        x           = self.norm_entrada(x)
        output, _   = self.lstm(x)           # (B, T, hidden*2)
        ultimo_paso = output[:, -1, :]       # (B, hidden*2)
        return self.cabeza(ultimo_paso)      # (B, n_clases)

    def predecir(self, x: torch.Tensor) -> tuple[str, float]:
        """Inferencia de un solo ejemplo. Retorna (nombre_clase, confianza)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            probs       = torch.softmax(self(x), dim=1)[0]
            conf, idx   = probs.max(dim=0)
        return self.clases[int(idx)], float(conf)

    # ── Factory / persistencia ────────────────────────────────────────────────

    @classmethod
    def crear(cls, tipo: str, feature_dim: int = 576, **kw) -> "LSTMEstados":
        """Crea un modelo para el tipo de máquina dado."""
        tipo_base = _normalizar_tipo(tipo)
        clases    = ESTADOS_POR_TIPO.get(tipo_base)
        if clases is None:
            raise ValueError(f"Tipo '{tipo}' desconocido. Opciones: {list(ESTADOS_POR_TIPO)}")
        return cls(feature_dim=feature_dim, clases=clases, **kw)

    @classmethod
    def cargar(cls, ruta: str | Path, device: str = "cpu") -> "LSTMEstados":
        """Carga un modelo guardado con .guardar()."""
        dev  = torch.device(device)
        ckpt = torch.load(str(ruta), map_location=dev, weights_only=False)
        m    = cls(
            feature_dim  = ckpt["feature_dim"],
            clases       = ckpt["clases"],
            hidden_size  = ckpt.get("hidden_size", HIDDEN_SIZE),
            num_layers   = ckpt.get("num_layers",  NUM_LAYERS),
            dropout      = ckpt.get("dropout",     DROPOUT),
        )
        m.load_state_dict(ckpt["state_dict"])
        m.to(dev).eval()
        acc = ckpt.get("acc_val", 0.0)
        nombre = Path(ruta).name
        print(f"[LSTMEstados] Cargado: {nombre} | clases={ckpt['clases']} | acc_val={acc:.3f}")
        return m

    def guardar(self, ruta: str | Path, acc_val: float = 0.0) -> None:
        """Guarda el modelo con sus metadatos."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "feature_dim": self.feature_dim,
            "clases":      self.clases,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     DROPOUT,
            "state_dict":  self.state_dict(),
            "acc_val":     acc_val,
        }, str(ruta))
        print(f"[LSTMEstados] Guardado: {ruta} (acc_val={acc_val:.3f})")

    def _init_pesos(self) -> None:
        for nombre, p in self.named_parameters():
            if "weight_ih" in nombre:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in nombre:
                nn.init.orthogonal_(p.data)
            elif "bias" in nombre:
                nn.init.zeros_(p.data)

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.parameters()) / 1e3
        return (
            f"LSTMEstados(Bidireccional, layers={self.num_layers}, "
            f"hidden={self.hidden_size}, clases={self.clases}, params={params:.1f}k)"
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class DatasetSecuencias(torch.utils.data.Dataset):
    """
    Dataset para entrenar la LSTM a partir de secuencias .npz.

    Formato del .npz:
        X:      (N, seq_len, feature_dim)  float32
        y:      (N,)                       int64
        clases: array de strings

    Uso:
        ds = DatasetSecuencias.desde_npz("data/secuencias_excavadora.npz")
    """

    def __init__(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        clases:  list[str],
        augment: bool = False,
    ) -> None:
        self.x      = torch.from_numpy(X.astype(np.float32))
        self.y      = torch.from_numpy(y.astype(np.int64))
        self.clases = clases
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        if self.augment:
            # Ruido gaussiano leve → simula variación de iluminación
            x = x + torch.randn_like(x) * 0.02
        return x, self.y[idx]

    @classmethod
    def desde_npz(cls, ruta: str | Path, augment: bool = False) -> "DatasetSecuencias":
        data   = np.load(str(ruta), allow_pickle=True)
        clases = list(data["clases"])
        return cls(data["X"], data["y"], clases, augment=augment)

    def guardar(self, ruta: str | Path) -> None:
        np.savez_compressed(
            str(ruta),
            X=self.x.numpy(), y=self.y.numpy(),
            clases=np.array(self.clases),
        )
        print(f"[Dataset] Guardado: {ruta} ({len(self)} muestras)")


# ── Entrenador ────────────────────────────────────────────────────────────────

class EntrenadorLSTM:
    """
    Entrenamiento con Early Stopping y CosineAnnealingLR.

    Uso:
        modelo  = LSTMEstados.crear("excavadora")
        ds      = DatasetSecuencias.desde_npz("data/secuencias_excavadora.npz")
        trainer = EntrenadorLSTM(modelo, ds)
        trainer.entrenar(epochs=60, ruta_guardado="weights/lstm_excavadora.pt")
    """

    def __init__(
        self,
        modelo:        LSTMEstados,
        ds_train:      DatasetSecuencias,
        ds_val:        Optional[DatasetSecuencias] = None,
        lr:            float = 1e-3,
        batch_size:    int   = 32,
        device:        str   = "cpu",
    ) -> None:
        self.modelo     = modelo.to(torch.device(device))
        self.ds_train   = ds_train
        self.ds_val     = ds_val
        self.device     = torch.device(device)
        self.optimizer  = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                              self.optimizer, T_max=60, eta_min=1e-5)
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.batch_size = batch_size

    def entrenar(
        self,
        epochs:        int            = 60,
        patience:      int            = 12,
        ruta_guardado: str | Path | None = None,
    ) -> dict:
        """Entrena y retorna historial {train_loss, val_acc}."""
        loader_tr = torch.utils.data.DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        loader_val = (
            torch.utils.data.DataLoader(self.ds_val, batch_size=self.batch_size,
                                        shuffle=False, num_workers=0)
            if self.ds_val else None
        )
        hist = {"train_loss": [], "val_acc": []}
        mejor_acc  = 0.0
        sin_mejora = 0

        for epoch in range(1, epochs + 1):
            # ── Entrenamiento ──────────────────────────────────────────────────
            self.modelo.train()
            total_loss = 0.0
            for xb, yb in loader_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.modelo(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            loss_media = total_loss / len(loader_tr)
            hist["train_loss"].append(loss_media)

            # ── Validación ────────────────────────────────────────────────────
            if loader_val:
                acc = self._eval(loader_val)
                hist["val_acc"].append(acc)
                marca = "✓" if acc > mejor_acc else " "
                print(f"Epoch {epoch:3d}/{epochs} | loss={loss_media:.4f} | val_acc={acc:.3f} {marca}")
                if acc > mejor_acc:
                    mejor_acc  = acc
                    sin_mejora = 0
                    if ruta_guardado:
                        self.modelo.guardar(ruta_guardado, acc_val=acc)
                else:
                    sin_mejora += 1
                if sin_mejora >= patience:
                    print(f"[Early Stop] {patience} épocas sin mejora.")
                    break
            else:
                print(f"Epoch {epoch:3d}/{epochs} | loss={loss_media:.4f}")

        return hist

    def _eval(self, loader) -> float:
        self.modelo.eval()
        ok = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                ok    += (self.modelo(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        return ok / total if total else 0.0


# ── Helper ─────────────────────────────────────────────────────────────────────

def _normalizar_tipo(tipo: str) -> str:
    return _ALIAS.get(tipo.lower(), tipo.lower())
