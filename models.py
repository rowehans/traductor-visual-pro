"""
models.py — Modelos SQLAlchemy para la app.
- SQLite para desarrollo local
- PostgreSQL/Supabase para produccion con RLS
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship

db = SQLAlchemy()

# ─── Modelos ────────────────────────────────────────────────────

class User(db.Model):
    """Usuario del sistema. Mapea a auth.users en Supabase."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    supabase_uid = Column(String(255), unique=True, nullable=True)  # UUID de Supabase Auth
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")

class Project(db.Model):
    """Proyecto de traduccion (un comic/manga completo)."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    source_lang = Column(String(10), default="auto")
    target_lang = Column(String(10), default="en")
    file_type = Column(String(10))  # pdf, image
    total_pages = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="projects")
    pages = relationship("Page", back_populates="project", cascade="all, delete-orphan")

class Page(db.Model):
    """Pagina individual de un proyecto."""
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    original_image = Column(Text, nullable=True)        # Ruta o URL de la imagen original
    inpainted_image = Column(Text, nullable=True)       # Ruta o URL de la imagen inpaintada
    status = Column(String(50), default="pending")      # pending, processing, done, error
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="pages")
    blocks = relationship("TextBlock", back_populates="page", cascade="all, delete-orphan")

class TextBlock(db.Model):
    """Bloque de texto detectado en una pagina."""
    __tablename__ = "text_blocks"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    w = Column(Integer, nullable=False)
    h = Column(Integer, nullable=False)
    source_text = Column(Text, nullable=True)
    translated_text = Column(Text, nullable=True)
    confidence = Column(Integer, default=0)
    font_size = Column(Integer, default=12)
    text_color = Column(String(20), default="#ffffff")
    bg_color = Column(String(20), default="#000000")
    polygon = Column(JSON, nullable=True)               # Poligono exacto [[x1,y1],...]
    is_edited = Column(Boolean, default=False)          # Si el usuario edito manualmente
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    page = relationship("Page", back_populates="blocks")


# ─── Inicializacion ─────────────────────────────────────────────

def init_db(app):
    """Configura la base de datos segun el entorno."""
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        # Produccion: PostgreSQL / Supabase
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    else:
        # Desarrollo: SQLite local
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
    def create(user_id: int, name: str, **kwargs) -> Project:
        p = Project(user_id=user_id, name=name, **kwargs)
        db.session.add(p)
        db.session.commit()
        return p

    @staticmethod
    def get_by_user(user_id: int) -> list[Project]:
        return Project.query.filter_by(user_id=user_id).order_by(Project.updated_at.desc()).all()

    @staticmethod
    def get_by_id(project_id: int, user_id: int) -> Project | None:
        return Project.query.filter_by(id=project_id, user_id=user_id).first()

    @staticmethod
    def delete(project_id: int, user_id: int) -> bool:
        p = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if p:
            db.session.delete(p)
            db.session.commit()
            return True
        return False


class PageRepository:
    """Acceso a datos de paginas."""

    @staticmethod
    def create(project_id: int, page_number: int, **kwargs) -> Page:
        p = Page(project_id=project_id, page_number=page_number, **kwargs)
        db.session.add(p)
        db.session.commit()
        return p

    @staticmethod
    def get_by_project(project_id: int) -> list[Page]:
        return Page.query.filter_by(project_id=project_id).order_by(Page.page_number).all()

    @staticmethod
    def get_by_id(page_id: int) -> Page | None:
        return Page.query.get(page_id)


class TextBlockRepository:
    """Acceso a datos de bloques de texto."""

    @staticmethod
    def bulk_save(page_id: int, blocks: list[dict]) -> list[TextBlock]:
        """Guarda o actualiza bloques de texto para una pagina."""
        # Eliminar bloques existentes
        TextBlock.query.filter_by(page_id=page_id).delete()
        # Insertar nuevos
        objs = []
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