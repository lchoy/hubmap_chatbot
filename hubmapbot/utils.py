from functools import lru_cache
from typing import List

from langchain.schema import Document

from hubmapbot import ENCODING


def num_tokens_from_string(string: str) -> int:
    num_tokens = len(ENCODING.encode(string))
    return num_tokens


def reduce_tokens_below_limit(max_tokens_limit, docs: List[Document]) -> List[Document]:
    num_docs = len(docs)

    tokens = [
        num_tokens_from_string(doc.page_content)
        for doc in docs
    ]
    token_count = sum(tokens[:num_docs])
    while token_count > max_tokens_limit:
        num_docs -= 1
        token_count -= tokens[num_docs]

    return docs[:num_docs]


def kwargs_to_dict(**kwargs):
    return kwargs


@lru_cache(maxsize=1)
def is_running_in_notebook():
    try:
        get_ipython().config
        return True
    except NameError:
        return False
