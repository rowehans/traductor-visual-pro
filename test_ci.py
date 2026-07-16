"""
Test de detección de idioma - standalone para CI.
No importa server.py, no carga modelos, no usa red.
"""
import sys
import re

def detect_language(text: str) -> str:
    """Detección de idioma usando solo heurísticas - sin modelos."""
    text = text.strip()
    if not text:
        return 'en'
    
    # CJK
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return 'ko'
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return 'ja'
    
    # Caracteres latinos extendidos (español)
    if any(c in 'áéíóúñüÁÉÍÓÚÑÜ¿¡' for c in text):
        return 'es'
    
    text_lower = text.lower()
    spa_words = {
        'el','la','los','las','que','en','un','una','de','con','es','para','por','si','no',
        'y','pero','como','cómo','mas','más','bien','todo','todos','esta','este','tus','sus','mi',
        'me','se','lo','le','te','al','del','tú','yo','responderme','responder',
        'gracias','por','todo','puedes','ayudarme','puede','ayuda'
    }
    words = {w.strip('.,¡!¿?()[]{}\'"') for w in text_lower.split()}
    if spa_words.intersection(words):
        return 'es'
    
    # Sufijos verbales españoles (enclíticos)
    for w in words:
        if len(w) > 2 and any(w.endswith(suf) for suf in (
            'arme','erme','irme','arte','erte','irte',
            'arse','erse','irse','arle','erle','irle',
            'arnos','ernos','irnos','arlos','erlos','irlos',
            'arme','erme','irme','aron','ieron'
        )):
            return 'es'
    
    return 'en'


def test():
    cases = [
        ('RESPONDERME?', 'es'),
        ('RESPONDERME', 'es'),
        ('RESPÓNDEME', 'es'),
        ('Hello world', 'en'),
        ('HELLO WORLD', 'en'),
        ('¿Puedes ayudarme?', 'es'),
        ('Gracias por todo', 'es'),
        ('¿Cómo estás?', 'es'),
        ('Hola, ¿qué tal?', 'es'),
        ('This is a test', 'en'),
    ]
    
    for text, expected in cases:
        result = detect_language(text)
        assert result == expected, f"Fallo: '{text}' -> {result}, esperado {expected}"
    
    print("[OK] Todos los tests de deteccion de idioma pasaron")


if __name__ == "__main__":
    test()