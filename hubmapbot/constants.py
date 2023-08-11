
from enum import Enum


class OpenAIModels(str, Enum):
    GPT_4 = "GPT-4"
    GPT_3_5 = "GPT-3.5"


class Modes(str, Enum):
    AUTO = "auto"
    GENERAL = "general"
    SDK = "sdk"
    INGEST = "ingest"
    # SEARCH = "search"
    SEARCH_DATASET = "search_dataset"
    SEARCH_SAMPLE = "search_sample"
    SEARCH_DONOR = "search_donor"
    OTHER = "other"

class Roles(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"