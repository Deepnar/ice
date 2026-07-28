"""C12 D10: text extraction behind a swappable ENGINE.

Open WebUI does not OCR in-process either — it exposes a content-extraction
engine setting whose non-default options (Apache Tika, Docling) run as their
own Docker containers and do the OCR there (Tika's image bundles Tesseract;
Docling exposes do_ocr / ocr_engine / ocr_lang). ICE copies that shape for the
same reason: Track F's end state is one installer, and a `tesseract` system
binary in setup.sh fights it.

So this module is the seam and nothing more. `builtin` (the default) is the
pure-python parser set in `parsers.py`, which extracts embedded text and
refuses scans loudly. Adding OCR later is a new branch here plus a compose
service — never a redesign of the pipeline above it.
"""

import structlog

from src.api.config import settings
from src.ingestion.documents import parsers

logger = structlog.get_logger("ice.ingestion.documents.extract")

ENGINES = ("builtin", "tika", "docling")

_DEFERRED = (
    "the '{engine}' extraction engine is not implemented yet — it is the Track F "
    "OCR item (run Tika or Docling as a container, point "
    "settings.document_extraction_engine at it). Use 'builtin' meanwhile.")


def current_engine() -> str:
    engine = getattr(settings, "document_extraction_engine", "builtin")
    if engine not in ENGINES:
        logger.warning("unknown_extraction_engine", engine=engine,
                       falling_back_to="builtin")
        return "builtin"
    return engine


def extract(path: str, file_type: str = None) -> parsers.DocumentParse:
    """File → DocumentParse via the configured engine."""
    engine = current_engine()
    if engine != "builtin":
        raise NotImplementedError(_DEFERRED.format(engine=engine))
    return parsers.parse_file(path, file_type)
