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
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
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
    confidence: Any = Column(Integer, default=0)
    font_size: Any = Column(Integer, default=12)
    text_color: Any = Column(String(20), default="#ffffff")
    bg_color: Any = Column(String(20), default="#000000")
    polygon: Any = Column(JSON, nullable=True)
    is_edited: Any = Column(Boolean, default=False)
    created_at: Any = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    page: Any = relationship("Page", back_populates="blocks")


# ─── Inicializacion ─────────────────────────────────────────────

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
        p = Project.query.filter_by(id=project_id, user_id=user_id).first()  # type: ignore[no-any-return]
        if p:
            db.session.delete(p)
            db.session.commit()
            return True
        return False


class PageRepository:
    """Acceso a datos de paginas."""

    @staticmethod
    def create(project_id: int, page_number: int, **kwargs: Any) -> Page:
        p = Page(project_id=project_id, page_number=page_number, **kwargs)
        db.session.add(p)
        db.session.commit()
        return p

    @staticmethod
    def get_by_project(project_id: int) -> list[Page]:
        return Page.query.filter_by(project_id=project_id).order_by(Page.page_number).all()  # type: ignore[no-any-return]

    @staticmethod
    def get_by_id(page_id: int) -> Page | None:
        return Page.query.get(page_id)  # type: ignore[no-any-return]


class TextBlockRepository:
    """Acceso a datos de bloques de texto."""

    @staticmethod
    def bulk_save(page_id: int, blocks: list[dict[str, Any]]) -> list[TextBlock]:
        TextBlock.query.filter_by(page_id=page_id).delete()  # type: ignore[no-any-return]
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
