"""Regresiones de tipos persistidos de los bloques OCR."""

from flask import Flask
import pytest
from sqlalchemy import Float, text

from models import (
    Page,
    PageRepository,
    Project,
    TextBlock,
    TextBlockRepository,
    User,
    db,
    _migrate_sqlite_confidence_type,
)


def test_confidence_de_textblock_admite_decimal():
    """La confianza OCR 0.0-1.0 no debe persistirse como entero."""
    assert isinstance(TextBlock.__table__.c.confidence.type, Float)


def test_page_repository_aplica_aislamiento_por_usuario():
    """Un usuario no puede enumerar ni leer páginas de otro proyecto."""
    app = Flask("test-models-isolation")
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)

    with app.app_context():
        db.create_all()
        owner = User(username="owner", email="owner@example.test")
        other = User(username="other", email="other@example.test")
        db.session.add_all([owner, other])
        db.session.flush()
        project = Project(user_id=owner.id, name="privado")
        db.session.add(project)
        db.session.flush()
        page = Page(project_id=project.id, page_number=1)
        db.session.add(page)
        db.session.commit()

        assert PageRepository.get_by_project(project.id, owner.id) == [page]
        assert PageRepository.get_by_project(project.id, other.id) == []
        assert PageRepository.get_by_id(page.id, owner.id) == page
        assert PageRepository.get_by_id(page.id, other.id) is None

        with pytest.raises(ValueError, match="no pertenece"):
            PageRepository.create(project.id, 2, other.id)
        with pytest.raises(ValueError, match="no pertenece"):
            TextBlockRepository.bulk_save(page.id, other.id, [])


def test_sqlite_migra_confidence_integer_a_float_sin_perder_datos(tmp_path):
    """Una DB creada antes del refactor conserva precisión tras arrancar."""
    app = Flask("test-models-migration")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'legacy.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)

    with app.app_context():
        db.create_all()
        db.session.execute(text("DROP TABLE text_blocks"))
        db.session.execute(text("""
            CREATE TABLE text_blocks (
                id INTEGER PRIMARY KEY,
                page_id INTEGER NOT NULL,
                x INTEGER NOT NULL, y INTEGER NOT NULL,
                w INTEGER NOT NULL, h INTEGER NOT NULL,
                source_text TEXT, translated_text TEXT,
                confidence INTEGER DEFAULT 0,
                font_size INTEGER DEFAULT 12,
                text_color VARCHAR(20) DEFAULT '#ffffff',
                bg_color VARCHAR(20) DEFAULT '#000000',
                polygon JSON, is_edited BOOLEAN DEFAULT 0,
                created_at DATETIME
            )
        """))
        db.session.execute(text(
            "INSERT INTO text_blocks "
            "(id, page_id, x, y, w, h, source_text, confidence) "
            "VALUES (1, 1, 2, 3, 4, 5, 'OCR', 0.85)"
        ))
        db.session.commit()

        _migrate_sqlite_confidence_type()

        info = db.session.execute(text("PRAGMA table_info(text_blocks)"))
        confidence_type = next(row[2] for row in info if row[1] == "confidence")
        value = db.session.execute(text(
            "SELECT confidence FROM text_blocks WHERE id = 1"
        )).scalar_one()

        assert confidence_type.upper() in {"FLOAT", "REAL", "DOUBLE"}
        assert value == pytest.approx(0.85)
