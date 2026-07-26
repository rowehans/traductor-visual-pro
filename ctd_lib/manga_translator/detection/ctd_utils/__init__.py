# __init__.py minimo para ctd_utils/ — solo exporta lo que usa ocr_ctd_fallback.py.
# textmask.py fue omitido intencionalmente (requiere manga_translator.utils que
# no esta disponible en este subset). ocr_ctd_fallback.py no lo necesita.
from .basemodel import TextDetBase, TextDetBaseDNN
from .utils.yolov5_utils import non_max_suppression
from .utils.db_utils import SegDetectorRepresenter
from .utils.imgproc_utils import letterbox
