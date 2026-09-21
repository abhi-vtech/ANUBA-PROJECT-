"""KDS ticket reading via the kds-ocr project.

The KDS screen is read by ``anuba-technologies/kds-ocr`` running as a child
process; this package consumes what it emits and presents it to the
production pipeline as a ticket source.

    reader.py     launches and supervises the kds-ocr process
    emissions.py  follows its recipe stream and validates each record
    mapping.py    its ingredient names and units -> our zones/classes, as counts
    journey.py    the full life of a ticket, for explaining a WRONG verdict
    client.py     FIFO ticket source, live requirement updates, bump verdicts

The verdict trigger is kds-ocr's explicit ``bumped`` emission, so nothing here
re-derives "the card disappeared".
"""

from src.kdsocr.client import KdsOcrClient
from src.kdsocr.emissions import APPEARED, BUMPED, UPDATED, RecipeEmission
from src.kdsocr.mapping import IngredientMapper, as_count
from src.kdsocr.reader import KdsOcrReader, ReaderConfig

__all__ = [
    "APPEARED",
    "BUMPED",
    "IngredientMapper",
    "KdsOcrClient",
    "KdsOcrReader",
    "ReaderConfig",
    "RecipeEmission",
    "UPDATED",
    "as_count",
]
