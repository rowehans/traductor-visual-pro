"""
analisis_calidad.py — Auditoría de calidad de traducción del PDF completo.
Analiza los 261 pares traducidos y los categoriza por calidad.
"""
import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('resultados_progreso.json','r',encoding='utf-8') as f:
    cp = json.load(f)

# ── Spanish common words (short, all-caps frequent in manga) ──
SHORT_SPANISH = {
    'EL','LA','LOS','LAS','UN','UNA','UNOS','UNAS',
    'DEL','CON','POR','QUE','QUÉ','SER','ESTA','ESTE',
    'ES','NO','SI','YA','PERO','MAS','MÁS','SUS','ERA',
    'HAN','LES','LE','TUS','NOS','SON','AL','MI','TU',
    'SE','TE','ME','LO','LA','LE','DE','EN','A','Y','O',
    'NI','DOY','DA','VA','VE','FUE','ERA','IR','HAY',
    'HE','HA','HAS','SOY','ERES','SOMOS','SON','SEA',
    'TAN','TAL','AH','OH','EH','AY','OK','BIEN','MAL',
    'CADA','TODO','OTRO','OTRA','GRAN','SIN','SOBRE',
    'TIPO','ALGO','NADA','ALGUIEN','NADIE','SIEMPRE',
    'NUNCA','TAMBIEN','TAMBIÉN','SOLO','SÓLO','MUY',
    'TANTO','NADA','VALE','LISTO','CLARO','CIERTO',
    'BUENO','MALO','COMO','CÓMO','DONDE','DÓNDE',
    'CUANDO','CUÁNDO','QUIEN','QUIÉN','AHORA','HOY',
    'AYER','MAÑANA','NUNCA','SIEMPRE','LUEGO','DESPUES',
    'DESPUÉS','ANTES','DURANTE','HASTA','DESDE','ENTRE',
    'SEGUN','SEGÚN','CONTRA','HACIA','PARA','POR',
}
# Common English words that might appear in manga
COMMON_ENGLISH = {
    'the','and','for','you','with','this','that','from',
    'have','not','but','are','was','his','her','all','can',
    'out','has','its','say','who','get','she','new','now',
    'how','one','two','three','four','five','six','seven',
    'eight','nine','ten','yes','no','ok','okay','hello',
    'good','bad','big','small','great','well','very','too',
}
CJK_RX = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
SPANISH_ACCENTS = set('áéíóúñüÁÉÍÓÚÑÜ¿¡')

def has_cjk(t): return bool(CJK_RX.search(t))
def has_accent(t): return any(c in SPANISH_ACCENTS for c in t)
def is_ocr_garbage(t):
    """Check if OCR output is clearly garbage (not valid text)."""
    stripped = t.strip()
    if not stripped: return False
    # All digits
    if re.match(r'^[\d\s.,;:\'\"\-_~]+$', stripped):
        return True
    # Mixed number-letter-noise patterns
    garbage_pats = [r'\d{4,}', r'\d[A-Z]{5,}', r'[A-Z]{6,}\d', r'\d{2,}[A-Z]{2,}\d{2,}']
    if any(re.search(p, t) for p in garbage_pats):
        return True
    # More than 40% digits
    digits = sum(1 for c in t if c.isdigit())
    if digits > 0 and digits / max(len(t), 1) > 0.4:
        return True
    # Special char noise: @#$%^&*+= in the middle
    special = re.findall(r'[@#$%^&*+=<>\[\]{}|\\/]', t)
    if len(special) >= 3:
        return True
    return False

def is_onomatopoeia(t):
    """Check if text is a sound effect / onomatopoeia that should stay untranslated."""
    t_stripped = t.strip(' \'"\u00a1\u00bf!?.,;:~-_()')
    t_upper = t_stripped.upper()
    if not t_upper: return False
    if has_cjk(t): return True
    if not t_upper.isalpha(): return False  # Must be purely letters
    if len(t_upper) < 3: return False  # Too short to be SFX (NO, SI, YA etc are dialogue)
    if len(t_upper) > 8: return False  # Too long for typical SFX
    if t_upper in SHORT_SPANISH: return False  # It's a Spanish word, not SFX
    
    # Known Korean/Japanese/English SFX romanizations
    known_sfx = {
        'BOOM','PUM','PUM','ZAS','CRASH','CLICK','PLOP','TOC','RING',
        'FLASH','BOING','POW','BANG','SMASH','SPLASH','BUMP','THUD',
        'WHAM','BUH','BOF','GRRR','GRR','BAAA','CLANG','SNIFF','GROAN',
        'SLAM','BEEP','WOOSH','NYOOM','FWOOOSH','KABOOM','KABOOM',
        'RUMBLE','SQUEAK','TWITTER','WHIR','ZOOM','VROOM','SCREECH',
        'GROWL','HOWL','SNAP','CRACKLE','POP','FIZZ','HISS','BUZZ',
        'WHISTLE','CHIME','DING','DONG','SPLAT','SQUISH','OOZE',
        'WHOOSH','AH','HUH','HEH','HAH','HMPH','PSST','SHH','SHH',
        'GASP','PANT','PHEW','WHEW','SIGH','UFF','UF','OH','OW',
        'OUCH','AY','OY','EH','BAH','MEH','BLEH','ACK','EEK','Eek','EEK',
        'TAP','TAP','KNOCK','KNOCK','RAP','RAP','PIT','PAT','PIT','PAT',
    }
    if t_upper in known_sfx: return True
    
    # Repeated letter pattern typical of SFX: GRRRR, BOOOOOOOM (3+ same letters)
    if re.search(r'(.)\1{2,}', t_upper):
        return True
    
    return False  # No clasificar automaticamente mayusculas sueltas como SFX;
                   # esos casos caen a UNTRANSLATED para revision manual

def quality(src, tgt):
    """Classify a translation pair into a quality category."""
    # Step 1: If source is OCR garbage, mark it regardless
    src_is_garbage = is_ocr_garbage(src)
    
    if src == tgt:
        if src_is_garbage:
            return "OCR_GARBAGE"
        if is_onomatopoeia(src):
            return "ONOMATOPOEIA"
        if is_english(src):
            return "ENGLISH"
        return "UNTRANSLATED"
    
    # Check if cleanup equals
    src_clean = src.strip(' \'"\u00a1\u00bf!?.,;:~-_()').upper()
    tgt_clean = tgt.strip(' \'"\u00a1\u00bf!?.,;:~-_()').upper()
    if src_clean == tgt_clean:
        if src_is_garbage:
            return "OCR_GARBAGE"
        return "UNTRANSLATED"
    
    if src_is_garbage:
        return "OCR_GARBAGE"
    
    # If translation recovered something from garbaged OCR, it's OCR_NOISY
    weird_chars = sum(1 for c in src if c.isdigit() or c in '@#$%^&*+=<>[]{}|\\/')
    if weird_chars >= 3:
        return "OCR_NOISY"
    
    # Natural vs literal: heuristic based on target naturalness
    # If target has contractions (don't, isn't, I'm), articles used naturally, etc
    natural_markers = ["'s", "n't", "'m", "'re", "'ve", "'ll", "the ", " a ", " an "]
    has_natural = any(m in tgt.lower() for m in natural_markers)
    
    if has_natural:
        return "BUENA"
    
    # Long translations with comparable word count = likely literal
    src_words = src.split()
    tgt_words = tgt.split()
    if len(src_words) >= 3 and len(tgt_words) >= 3:
        ratio = len(tgt_words) / max(len(src_words), 1)
        if 0.5 < ratio < 1.5:  # similar word count
            return "LITERAL"
    
    # Default for short translations
    return "BUENA"

def is_english(t):
    """Check if text appears to be English already."""
    words = t.lower().split()
    eng_count = sum(1 for w in words for ew in COMMON_ENGLISH if w.strip('.,;:\'"!?') == ew)
    if eng_count >= 2 and not has_accent(t):
        return True
    return False

# ── Process all pairs ──────────────────────────────────────────
categories = {
    "BUENA":        [],
    "LITERAL":      [],
    "OCR_NOISY":    [],
    "OCR_GARBAGE":  [],
    "UNTRANSLATED": [],
    "ONOMATOPOEIA": [],
    "ENGLISH":      [],
}

all_pairs = []
for r in cp['results']:
    pg = r['page']
    for t in r.get('texts', []):
        src = t.get('src', '')
        tgt = t.get('tgt', '')
        if not src and not tgt:
            continue
        cat = quality(src, tgt)
        all_pairs.append((pg, cat, src, tgt))
        categories[cat].append((pg, src, tgt))

# ── Print report ───────────────────────────────────────────────
DIV = "=" * 90
SDIV = "\u2500" * 80  # ─

print(DIV)
print("  INFORME DE CALIDAD DE TRADUCCION — " + str(len(all_pairs)) + " pares analizados")
print(DIV)
print()
header = f"  {'Categoria':<22s} {'Cant.':>6s} {'Porc.':>7s}"
print(header)
print("  " + SDIV[:38])
total = len(all_pairs)
for cat in ["BUENA", "LITERAL", "OCR_NOISY", "OCR_GARBAGE", "ONOMATOPOEIA", "ENGLISH", "UNTRANSLATED"]:
    count = len(categories[cat])
    pct = count / total * 100 if total else 0
    icon = {"BUENA": "\u2705", "LITERAL": "\U0001f4d6", "OCR_NOISY": "\u26a0\ufe0f",
            "OCR_GARBAGE": "\u274c", "ONOMATOPOEIA": "\U0001f50a",
            "ENGLISH": "\U0001f310", "UNTRANSLATED": "\u2753"}[cat]
    print(f"  {icon} {cat:<20s} {count:>6d} {pct:>6.1f}%")
print()
print(f"  {'TOTAL':<22s} {total:>6d} {100:>6.1f}%")
print()

# ── Detailed samples ───────────────────────────────────────────
print()
print(DIV)
print("  EJEMPLOS POR CATEGORIA")
print(DIV)

print()
print("  \u2705 BUENAS (traducciones correctas y naturales)")
print("  " + SDIV)
for pg, src, tgt in categories["BUENA"][:10]:
    print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"  ->  \"{tgt[:45]:45s}\"")

print()
print("  \U0001f4d6 LITERALES (correctas pero palabra-por-palabra)")
print("  " + SDIV)
for pg, src, tgt in categories["LITERAL"][:10]:
    print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"  ->  \"{tgt[:45]:45s}\"")

if categories["OCR_NOISY"]:
    print()
    print("  \u26a0\ufe0f  OCR RUIDOSO (OCR alucino pero traduccion se recupera)")
    print("  " + SDIV)
    for pg, src, tgt in categories["OCR_NOISY"][:10]:
        print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"  ->  \"{tgt[:45]:45s}\"")

print()
print("  \u274c OCR BASURA (OCR tan ruidoso que la traduccion es inservible)")
print("  " + SDIV)
for pg, src, tgt in categories["OCR_GARBAGE"][:15]:
    print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"  ->  \"{tgt[:45]:45s}\"")

if categories["ONOMATOPOEIA"]:
    print()
    print("  \U0001f50a ONOMATOPEYAS (efectos de sonido, correctamente sin traducir)")
    print("  " + SDIV)
    for pg, src, tgt in categories["ONOMATOPOEIA"][:10]:
        print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"")

if categories["ENGLISH"]:
    print()
    print("  \U0001f310 YA EN INGLES (texto ya estaba en ingles)")
    print("  " + SDIV)
    for pg, src, tgt in categories["ENGLISH"][:10]:
        print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"")

if categories["UNTRANSLATED"]:
    print()
    print("  \u2753 SIN TRADUCIR (texto quedo igual sin razon aparente)")
    print("  " + SDIV)
    for pg, src, tgt in categories["UNTRANSLATED"][:10]:
        print(f"     Pg {pg:3d}: \"{src[:45]:45s}\"")

# ── Summary ────────────────────────────────────────────────────
print()
print(DIV)
print("  RESUMEN DE CALIDAD")
print(DIV)
print()
good = len(categories["BUENA"]) + len(categories["LITERAL"]) + len(categories["OCR_NOISY"])
sfx = len(categories["ONOMATOPOEIA"])
garbage = len(categories["OCR_GARBAGE"])
other = len(categories["ENGLISH"]) + len(categories["UNTRANSLATED"])

print(f"  \U0001f7e2 Aceptables (BUENA + LITERAL + OCR_NOISY):   {good:3d}/{total} ({good/total*100:5.1f}%)")
print(f"  \U0001f50a Onomatopeyas correctamente ignoradas:     {sfx:3d}/{total} ({sfx/total*100:5.1f}%)")
print(f"  \U0001f534 Traducciones basura (OCR_GARBAGE):         {garbage:3d}/{total} ({garbage/total*100:5.1f}%)")
print(f"  \u26aa Otros (ingles + sin traducir):               {other:3d}/{total} ({other/total*100:5.1f}%)")
print()
acceptance = (good + sfx) / total * 100
print(f"  \U0001f3c6 Tasa de aceptacion global: {acceptance:.1f}%")
print(f"     (aceptamos: traducciones correctas + onomatopeyas)")
print()

# ── All garbage list ──────────────────────────────────────────
if categories["OCR_GARBAGE"]:
    print()
    print("  LISTA COMPLETA DE TRADUCCIONES BASURA:")
    print("  " + SDIV)
    for pg, src, tgt in categories["OCR_GARBAGE"]:
        print(f"     Pg {pg:3d}: \"{src[:50]:50s}\"  ->  \"{tgt[:50]:50s}\"")

# ── Summary by page ────────────────────────────────────────────
print()
print(DIV)
print("  PAGINAS CON PEOR CALIDAD")
print(DIV)
print()
# Rank pages by number of garbage blocks
from collections import Counter
garbage_by_page = Counter()
for pg, cat, src, tgt in all_pairs:
    if cat == "OCR_GARBAGE":
        garbage_by_page[pg] += 1
if garbage_by_page:
    for pg, count in garbage_by_page.most_common(10):
        print(f"     Pag {pg:3d}: {count} bloque(s) basura")
