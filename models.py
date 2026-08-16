"""
models.py — Modelos SQLAlchemy para la app.
- SQLite para desarrollo local
- PostgreSQL/Supabase para produccion con RLS
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Boolean,
    Float,
    MetaData,
    Table,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ─── Modelos ────────────────────────────────────────────────────

class Base(DeclarativeBase):
    __allow_unmapped__ = True  # SQLAlchemy 2.0: permite anotaciones tipo : Any = Column(...)


db: SQLAlchemy = SQLAlchemy(model_class=Base)


class User(db.Model):  # type: ignore[misc, name-defined]
    """Usuario del sistema. Mapea a auth.users en Supabase."""
    __tablename__ = "users"

    id: Any = Column(Integer, primary_key=True)
    supabase_uid: Any = Column(String(255), unique=True, nullable=True)
    username: Any = Column(String(100), unique=True, nullable=False)
    email: Any = Column(String(255), unique=True, nullable=False)
    created_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active: Any = Column(Boolean, default=True)

    projects: Any = relationship("Project", back_populates="owner", cascade="all, delete-orphan")


class Project(db.Model):  # type: ignore[misc, name-defined]
    """Proyecto de traduccion (un comic/manga completo)."""
    __tablename__ = "projects"

    id: Any = Column(Integer, primary_key=True)
    user_id: Any = Column(Integer, ForeignKey("users.id"), nullable=False)
    name: Any = Column(String(255), nullable=False)
    source_lang: Any = Column(String(10), default="auto")
    target_lang: Any = Column(String(10), default="en")
    file_type: Any = Column(String(10))
    total_pages: Any = Column(Integer, default=0)
    created_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    owner: Any = relationship("User", back_populates="projects")
    pages: Any = relationship("Page", back_populates="project", cascade="all, delete-orphan")


class Page(db.Model):  # type: ignore[misc, name-defined]
    """Pagina individual de un proyecto."""
    __tablename__ = "pages"

    id: Any = Column(Integer, primary_key=True)
    project_id: Any = Column(Integer, ForeignKey("projects.id"), nullable=False)
    page_number: Any = Column(Integer, nullable=False)
    original_image: Any = Column(Text, nullable=True)
    inpainted_image: Any = Column(Text, nullable=True)
    status: Any = Column(String(50), default="pending")
    created_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project: Any = relationship("Project", back_populates="pages")
    blocks: Any = relationship("TextBlock", back_populates="page", cascade="all, delete-orphan")


class TextBlock(db.Model):  # type: ignore[misc, name-defined]
    """Bloque de texto detectado en una pagina."""
    __tablename__ = "text_blocks"

    id: Any = Column(Integer, primary_key=True)
    page_id: Any = Column(Integer, ForeignKey("pages.id"), nullable=False)
    x: Any = Column(Integer, nullable=False)
    y: Any = Column(Integer, nullable=False)
    w: Any = Column(Integer, nullable=False)
    h: Any = Column(Integer, nullable=False)
    source_text: Any = Column(Text, nullable=True)
    translated_text: Any = Column(Text, nullable=True)
    # OCR devuelve una confianza continua [0.0, 1.0]; Integer la truncaba
    # al persistir y destruÃ­a la seÃ±al usada para revisar resultados.
    confidence: Any = Column(Float, default=0.0)
    font_size: Any = Column(Integer, default=12)
    text_color: Any = Column(String(20), default="#ffffff")
    bg_color: Any = Column(String(20), default="#000000")
    polygon: Any = Column(JSON, nullable=True)
    is_edited: Any = Column(Boolean, default=False)
    created_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    page: Any = relationship("Page", back_populates="blocks")


# ─── Inicializacion ─────────────────────────────────────────────

def _migrate_sqlite_confidence_type() -> None:
    """Actualiza la columna legacy confidence sin perder filas.

    db.create_all() crea tablas nuevas, pero no modifica el tipo de las
    columnas existentes. La base local puede venir de una version donde la
    confianza OCR era INTEGER y debe reconstruirse solo esa tabla.
    """
    if db.engine.dialect.name != "sqlite":
        return

    legacy_name = "text_blocks_legacy_confidence"
    model_columns = [column.name for column in TextBlock.__table__.columns]

    def table_exists(connection: Any, name: str) -> bool:
        return connection.execute(
            text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = :name"
            ),
            {"name": name},
        ).first() is not None

    try:
        with db.engine.connect() as connection:
            table_info = connection.exec_driver_sql(
                "PRAGMA table_info(text_blocks)"
            ).fetchall()
            if not table_info:
                return

            legacy_columns = {str(row[1]) for row in table_info}
            confidence_type = next(
                (str(row[2] or "").upper() for row in table_info if row[1] == "confidence"),
                "",
            )
            if confidence_type in {"FLOAT", "REAL", "DOUBLE", "DOUBLE PRECISION"}:
                return

            # No reutilizar un backup abandonado: requiere inspeccion manual
            # y evita sobreescribir datos en una recuperacion posterior.
            if table_exists(connection, legacy_name):
                print(
                    "[db] Aviso: existe una tabla de migracion pendiente; "
                    "se conserva el esquema actual"
                )
                return

            missing_columns = [name for name in model_columns if name not in legacy_columns]
            if missing_columns:
                print(
                    "[db] Aviso: no se migra text_blocks; faltan columnas "
                    + ", ".join(missing_columns)
                )
                return

            renamed = False
            # Las consultas de inspeccion usan autobegin en SQLAlchemy 2.
            # Cerrarlo antes de iniciar la migracion DDL explicita.
            connection.commit()
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            connection.commit()
            transaction = connection.begin()
            try:
                connection.execute(
                    text(
                        "ALTER TABLE text_blocks "
                        "RENAME TO text_blocks_legacy_confidence"
                    )
                )
                renamed = True
                db.metadata.create_all(
                    bind=connection,
                    tables=[TextBlock.__table__],
                )
                legacy_table = Table(
                    legacy_name,
                    MetaData(),
                    autoload_with=connection,
                )
                connection.execute(
                    TextBlock.__table__.insert().from_select(
                        model_columns,
                        select(
                            *(legacy_table.c[name] for name in model_columns)
                        ),
                    )
                )
                legacy_table.drop(connection)
                transaction.commit()
                print(
                    "[db] Migracion aplicada: text_blocks.confidence "
                    "INTEGER -> FLOAT"
                )
            except Exception:
                transaction.rollback()

                # SQLite suele revertir DDL dentro de la transaccion. Si el
                # driver deja el rename persistido, restaurar el backup antes
                # de continuar para no dejar una tabla nueva vacia.
                if renamed and table_exists(connection, legacy_name):
                    if table_exists(connection, "text_blocks"):
                        connection.execute(
                            text("DROP TABLE text_blocks")
                        )
                    connection.execute(
                        text(
                            "ALTER TABLE text_blocks_legacy_confidence "
                            "RENAME TO text_blocks"
                        )
                    )
                    connection.commit()
                raise
            finally:
                connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    except Exception as exc:
        print(f"[db] Aviso: no se pudo migrar confidence: {exc}")


def init_db(app: Flask) -> None:
    """Configura la base de datos segun el entorno."""
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    else:
        if getattr(sys, 'frozen', False):
            # En modo frozen (.exe), usar %LOCALAPPDATA% para persistencia
            localappdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            db_path = Path(localappdata) / "TraductorVisual" / "data" / "traductor.db"
        else:
            db_path = Path(__file__).parent / "data" / "traductor.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()
        _migrate_sqlite_confidence_type()

    print(f"[db] Base de datos lista: {app.config['SQLALCHEMY_DATABASE_URI']}")


# ─── Repositorios (Data Access Layer) ───────────────────────────

class ProjectRepository:
    """Acceso a datos de proyectos con aislamiento por usuario."""

    @staticmethod
    def create(user_id: int, name: str, **kwargs: Any) -> Project:
        p = Project(user_id=user_id, name=name, **kwargs)
        db.session.add(p)
        db.session.commit()
        return p

    @staticmethod
    def get_by_user(user_id: int) -> list[Project]:
        return Project.query.filter_by(user_id=user_id).order_by(Project.updated_at.desc()).all()  # type: ignore[no-any-return]

    @staticmethod
    def get_by_id(project_id: int, user_id: int) -> Project | None:
        return Project.query.filter_by(id=project_id, user_id=user_id).first()  # type: ignore[no-any-return]

    @staticmethod
    def delete(project_id: int, user_id: int) -> bool:
        p = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if p:
            db.session.delete(p)
            db.session.commit()
            return True
        return False


class PageRepository:
    """Acceso a datos de páginas con aislamiento por usuario."""

    @staticmethod
    def create(
        project_id: int,
        page_number: int,
        user_id: int,
        **kwargs: Any,
    ) -> Page:
        project = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if project is None:
            raise ValueError("El proyecto no pertenece al usuario")
        p = Page(project_id=project_id, page_number=page_number, **kwargs)
        db.session.add(p)
        db.session.commit()
        return p

    @staticmethod
    def get_by_project(project_id: int, user_id: int) -> list[Page]:
        return (  # type: ignore[no-any-return]
            Page.query
            .join(Project, Page.project_id == Project.id)
            .filter(Page.project_id == project_id, Project.user_id == user_id)
            .order_by(Page.page_number)
            .all()
        )

    @staticmethod
    def get_by_id(page_id: int, user_id: int) -> Page | None:
        return (  # type: ignore[no-any-return]
            Page.query
            .join(Project, Page.project_id == Project.id)
            .filter(Page.id == page_id, Project.user_id == user_id)
            .first()
        )


class TextBlockRepository:
    """Acceso a datos de bloques de texto con aislamiento por usuario."""

    @staticmethod
    def bulk_save(
        page_id: int,
        user_id: int,
        blocks: list[dict[str, Any]],
    ) -> list[TextBlock]:
        page = (
            Page.query
            .join(Project, Page.project_id == Project.id)
            .filter(Page.id == page_id, Project.user_id == user_id)
            .first()
        )
        if page is None:
            raise ValueError("La página no pertenece al usuario")
        TextBlock.query.filter_by(page_id=page_id).delete()
        objs: list[TextBlock] = []
        for b in blocks:
            objs.append(TextBlock(
                page_id=page_id,
                x=b.get("x", 0),
                y=b.get("y", 0),
                w=b.get("w", 0),
                h=b.get("h", 0),
                source_text=b.get("text", ""),
                translated_text=b.get("translated", ""),
                confidence=b.get("confidence", 0),
                font_size=b.get("fontSize", 12),
                text_color=b.get("textColor", "#ffffff"),
                bg_color=b.get("bgColor", "#000000"),
                polygon=b.get("polygon"),
            ))
        db.session.add_all(objs)
        db.session.commit()
        return objs
