"""
Capa de persistencia: gestión de conexión SQLite y repositorio de eventos.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, Evento


def get_engine(db_path: str):
    """Crea y retorna el engine SQLAlchemy para SQLite."""
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )


def init_db(db_path: str) -> None:
    """
    Crea el archivo SQLite y la tabla 'eventos' si no existen.
    Operación idempotente: llamar múltiples veces no produce errores.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def get_session_factory(db_path: str):
    """Retorna una SessionFactory configurada para la DB en db_path."""
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)


class RepositorioEventos:
    """
    Repositorio para operaciones CRUD sobre la tabla 'eventos'.

    Usa transacciones atómicas para inserción.
    Los errores de inserción se logean y no detienen la ejecución.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def insertar(self, evento: dict) -> None:
        """
        Inserta un evento en la tabla 'eventos' dentro de una transacción atómica.

        En caso de error: loguea [DB ERROR] con los datos del evento, hace rollback
        y NO relanza la excepción (el pipeline continúa).

        Args:
            evento: dict con claves {tipo_maquina, clase_id, confianza,
                    numero_fotograma, timestamp}
        """
        try:
            db_evento = Evento(**evento)
            self.session.add(db_evento)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"[DB ERROR] No se pudo insertar evento: {evento} | {e}")

    def obtener_todos(self) -> list[Evento]:
        """
        Retorna todos los eventos ordenados por timestamp ascendente.

        Returns:
            Lista de objetos Evento ordenados por timestamp ASC.
        """
        return (
            self.session.query(Evento)
            .order_by(Evento.timestamp.asc())
            .all()
        )

    def obtener_por_id(self, evento_id: int) -> Evento | None:
        """
        Retorna el evento con el id dado, o None si no existe.

        Args:
            evento_id: ID del evento a buscar.

        Returns:
            Objeto Evento o None.
        """
        return self.session.get(Evento, evento_id)
