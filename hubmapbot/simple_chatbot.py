import copy
from dataclasses import dataclass
from doctest import debug
from functools import lru_cache
import os
import shlex

from dotenv import load_dotenv


import re

import rich.console
import rich.markdown

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path

from collections import namedtuple

import openai
from enum import Enum

from langchain.docstore.document import Document
from typing import List, Any, Callable

import logging
import tiktoken

import pandas as pd

from hubmap_sdk import SearchSdk
import json
from collections import defaultdict

from hubmap_sdk.sdk_helper import HTTPException

import markdown_strings


load_dotenv(dotenv_path='.env')

logger = logging.getLogger(__name__)

# MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MODEL_NAME = "gpt-3.5-turbo-0613"

def chat_completion_default_kwargs(model_name):
    return {
        "model": model_name,
    }

ENCODING = tiktoken.encoding_for_model(MODEL_NAME)

SERVICE_URL = "https://search.api.hubmapconsortium.org/v3/"
SEARCHSDK_INSTANCE = SearchSdk(service_url=SERVICE_URL)

DEBUG_OUT_DIR = Path("debug_out")

data_path = Path("data")
github_path = Path("github")
persist_directory = Path("persist")

vectorstore_info = namedtuple("vectorstore_info", ["name", "path", "col"])

VECTORSTORE_INFOS = {
    "website": vectorstore_info("website", persist_directory / "chroma", "langchain"),
    "es_dataset": vectorstore_info("es_dataset", persist_directory / "chroma_es_dataset", "es_dataset"),
}


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


MODE_MAP = {
    "Auto 💬": Modes.AUTO,
    "Dataset Search 📊🔎": Modes.SEARCH_DATASET,
    # "Sample Search 🧫🔎": RetrieverOptions.SEARCH_SAMPLE,
    # "Donor Search 👨🔎": RetrieverOptions.SEARCH_DONOR,
    # "Other": RetrieverOptions.OTHER,
    "General Q&A 🤔": Modes.GENERAL,
}

MODEL_MAP = {
    "GPT-4": "gpt-4-0613",
    "GPT-3.5": "gpt-3.5-turbo-0613",
}


class Roles(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: Roles
    summary_content: str
    renderable_content: List[object]
    successful: bool = True
    num_results: int = 0



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


NO_PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": [],
}

JSON_MD = """```json
{json}
```"""

INITIAL_MESSAGE = "Hello, I am the HuBMAP Chatbot. How can I help you today?"

INVALID_API_KEY_MSG = "No valid OpenAI API key found. Please enter a valid API key. You can get one [here](https://platform.openai.com/account/api-keys)."


INITIAL_SUGGESTIONS_MAP = defaultdict(list)
INITIAL_SUGGESTIONS_MAP[Modes.AUTO] = [
    # "How do I filter datasets by creation date?",
    # "What does the 'name' field in donor objects mean?",
    # "Perform a search.",
    # "Search for datasets.",
    # "Search for donors.",
    # "Perform a dataset search and filter by donor age, where the donors are old.",
    "What is HuBMAP?",
    "What is the HuBMAP Data Portal?",
    "What are HuBMAP IDs?",
    "How do I download bulk data from hubmap?",
    "Search for spleen datasets published after 2021 with files ending in .tiff files larger than 5 gigabytes",
    "Find CODEX [Cytokit + SPRM] datasets from donors with a weight between 80-100 kg and a height between 140-180cm",
    "Find datasets from white males between 25-35 years old with a history of cardiac arrest and hypertension who passed from anoxia",
    "Find left kidney datasets with a donor profile index under 50, where the donor is under 30 years old and is blood type O, and died from anoxia",
]
INITIAL_SUGGESTIONS_MAP[Modes.GENERAL] = [
    "What is HuBMAP?",
    "What is the HuBMAP Data Portal?",
    "What are HuBMAP IDs?",
    "How do I download bulk data from hubmap?",
    # "Does HuBMAP provide Millitomes?",
]
INITIAL_SUGGESTIONS_MAP[Modes.SEARCH_DATASET] = [
    # "Search for datasets are derived from male donors with diabetes",
    # "Find datasets derived from white female donors who died from natural causes",
    # "Find datasets containing arrow files",
    # "Find datasets published after 2020",
    # "Find datasets from donors who smoke or have smoked with a BMI less than 40",
    # "Find datasets from donors with type A blood",
    # "Find datasets from african-american donors between 45-72",
    "Search for spleen datasets published after 2021 with files ending in .tiff files larger than 5 gigabytes",
    "Find CODEX [Cytokit + SPRM] datasets from donors with a weight between 80-100 kg and a height between 140-180cm",
    "Find datasets from white males between 25-35 years old with a history of cardiac arrest and hypertension who passed from anoxia",
    "Find left kidney datasets with a donor profile index under 50, where the donor is under 30 years old and is blood type O, and died from anoxia",
]


YOU_ARE = "You are a chatbot on the website for the Human BioMolecular Atlas Program (HuBMAP) research initiative. You are capable of answering questions about the initiative, how to use the HuBMAP sdks, or running searches for HuBMAP donors, samples, or datasets."

FIELD_DESCRIPTION_PROMPT = YOU_ARE+""" The following is a field in a HuBMAP dataset object. Given the field name and example values, describe what the field is used for in one sentence. Do not repeat the field name in your description.

Example:
donor.mapped_metadata.cause_of_death: "The cause of death of the donor."
donor.mapped_metadata.age_value: "The age of the donor at the time of death."

mapper_metadata.size, Example values:
54529
101744
99825
138588
95111
200033
27844
112596
92580
20194
33281
139520
18434
31760
23946
25168
88233
23270
"""

SIMPLE_ENGINE_PROMPT = YOU_ARE + """ You can use markdown in your response."""

CONDENSE_QUESTION_PROMPT = """The following is the chat history between a chatbot on the website for the Human BioMolecular Atlas Program (HuBMAP) research initiative. The chatbot is capable of answering general questions about information found on the portal website or what kinds of information is generally found in HuBMAP datasets. The chatbot is also capable of searching the HuBMAP database for specific donors, samples, or datasets that meet specific criteria. Given the chat history and follow up message, rephrase the user's message into standalone question or request that the chatbot would be capable of fulfilling without seeing any of the chat history. Make sure that all relevant information for answering the user's follow-up message is included in the standalone message. The follow up message may be unrelated to the previous chat history, in which case you can repeat the user's message.

Chat History:
{chat_history}


Follow Up User Message: {question}
Standalone question or request:"""

NEEDS_HISTORY_PROMPT = """
The following is the latest message between an assistant and a user. Do you need to see the chat history to answer the user's question? If so, call the "yes" function. Otherwise, call the "no" function.

User:
{question}
"""

# RETRIEVER_SELECT_PROMPT = YOU_ARE + """ Call the function that best describes the following user prompt. Do not respond to the user prompt.

# User prompt:
# {question}
# """


# RETRIEVER_SELECT_FUNCTIONS = [
#     {"name": "general", "description": "The user has a question about the HuBMAP project", "parameters": NO_PARAMETERS},
#     {"name": "sdk", "description": "The user has a question about using hubmap sdks", "parameters": NO_PARAMETERS},
#     {"name": "ingest", "description": "The user has a question about the Ingest process for new HuBMAP data", "parameters": NO_PARAMETERS},
#     {
#         "name": "search", 
#         "description": "The user has a specific search query in mind for either a HuBMAP dataset, sample, or donor", 
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "entity_type": {
#                     "type": "string",
#                     "description": "Either 'dataset', 'sample', or 'donor'",
#                 },
#             },
#             "required": ["entity_type"],
#         },
#     },
#     # {"name": "search_sample", "description": "Question about searching HuBMAP samples", "parameters": NO_PARAMETERS},
#     {"name": "about", "description": "The user has a question about you, the chatbot", "parameters": NO_PARAMETERS},
#     {"name": "other", "description": "The user prompt does not fall into any of the above categories", "parameters": NO_PARAMETERS},
# ]

# RETRIEVER_SELECT_FUNCTIONS = [
#     {"name": "general", "description": "Answer a question about the HuBMAP project", "parameters": NO_PARAMETERS},
#     {"name": "sdk", "description": "Answer a question about using hubmap sdks", "parameters": NO_PARAMETERS},
#     {"name": "ingest", "description": "Answer a question about the Ingest process for new HuBMAP data", "parameters": NO_PARAMETERS},
#     {
#         "name": "search", 
#         "description": "Construct a search query for HuBMAP datasets, samples, or donors that matches the user's search parameters", 
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "entity_type": {
#                     "type": "string",
#                     "description": "Either 'dataset', 'sample', or 'donor'",
#                 },
#             },
#             "required": ["entity_type"],
#         },
#     },
#     # {"name": "search_sample", "description": "Question about searching HuBMAP samples", "parameters": NO_PARAMETERS},
#     {"name": "about", "description": "Answer a question about you, the chatbot", "parameters": NO_PARAMETERS},
#     {"name": "other", "description": "Answer a question that does not fall into any of the above categories", "parameters": NO_PARAMETERS},
# ]


# RETRIEVER_SELECT_PROMPT = """Call the function that best describes the following user prompt.

# User prompt:
# {question}
# """


# RETRIEVER_SELECT_PROMPT = """
# Categorize the user's question. Do not respond to the user prompt.

# general: The user has a question about the HuBMAP project
# sdk: The user has a question about using hubmap sdks
# ingest: The user has a question about the Ingest process for new HuBMAP data
# search: The user is telling me to search the database of HuBMAP datasets, samples, or donors
# about_search: The user has a question related to searching for HuBMAP datasets, samples, or donors, but doesn't actually want a specific search performed
# about: The user has a question about you, the chatbot
# other: The user prompt does not fall into any of the above categories

# User prompt:
# {question}
# """
RETRIEVER_SELECT_PROMPT = """
Categorize the user's question. Do not respond to the user prompt.

general: The user has a question about the HuBMAP project
sdk: The user has a question about using hubmap sdks
ingest: The user has a question about the Ingest process for new HuBMAP data
search: The user specified an entirely self-contained search query for HuBMAP datasets, samples, or donors and mentioned both specific fields (such as age, ethnicity, creation date, type) and what values to filter those fields by (30 years old, asian, 2021, CODEX). 
about_search: The user has a question related to searching for HuBMAP datasets, samples, or donors, but didn't provide both specific fields and values to filter those fields by.
about: The user has a question about you, the chatbot
other: The user prompt does not fall into any of the above categories

User prompt:
{question}
"""

RETRIEVER_SELECT_FUNCTIONS = [
    # {"name": "general", "description": "Answer a question about the HuBMAP project", "parameters": NO_PARAMETERS},
    # {"name": "sdk", "description": "Answer a question about using hubmap sdks", "parameters": NO_PARAMETERS},
    # {"name": "ingest", "description": "Answer a question about the Ingest process for new HuBMAP data", "parameters": NO_PARAMETERS},
    # {
    #     "name": "search", 
    #     "description": "Construct a search query for HuBMAP datasets, samples, or donors that matches the user's search parameters", 
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "entity_type": {
    #                 "type": "string",
    #                 "description": "Either 'dataset', 'sample', or 'donor'",
    #             },
    #         },
    #         "required": ["entity_type"],
    #     },
    # },
    # # {"name": "search_sample", "description": "Question about searching HuBMAP samples", "parameters": NO_PARAMETERS},
    # {"name": "about", "description": "Answer a question about you, the chatbot", "parameters": NO_PARAMETERS},
    # {"name": "other", "description": "Answer a question that does not fall into any of the above categories", "parameters": NO_PARAMETERS},
    {"name": "categorize_question", "description": "Categorize the user's question",
     "parameters": {
         "type": "object",
         "properties": {
             "category": {
                 "type": "string",
                 "description": "Either 'general', 'sdk', 'ingest', 'search', 'about_search', 'about', or 'other'",
             },
             "entity_type": {
                "type": "string",
                "description": "Either 'dataset', 'sample', or 'donor', only required if category is 'search'",
             }
         },
         "required": ["category"],
     }},
]

# If the context doesn't provide the answer, call the “I don't know” function.

TEXT_ANSWER_PROMPT = """You are an assistant on the website for the Human BioMolecular Atlas Program (HuBMAP) research initiative. Use the following pieces of context to answer my question at the end. You can use markdown in your response.
If the context doesn't provide the answer, say "I couldn't find an answer to your question from official HuBMAP sources. Please try rephrasing your question, or try asking a different question."

{context}

My Question: {question}
Helpful Answer:
"""

TEXT_ANSWER_FUNCTIONS = [
    {"name": "dont_know", "description": "I don't know", "parameters": NO_PARAMETERS},
]

TEXT_ANSWER_FAIL_MSG = """Sorry, I couldn't find an answer to your question. Please try rephrasing your question, or try asking a different question."""

#  If it is not possible to construct a query that fulfills the user's search parameters using the fields below, apologize and explain why you could not create the query, and give suggestions for how the user can fix their query. You can use markdown.
# Include a markdown explanation of the query you constructed, the names of the fields you chose to use, and why. 

# ```text
# [your explanation of how you constructed the query or why you could not construct the query]
# ```
# Give me only the json code part of the answer. 
# Include a brief explanation of the query you constructed, the names of the fields you chose to use, and why.
# First, explain which fields you will use and how.
ES_SEARCH_PROMPT = """You are an assistant on the website for the Human BioMolecular Atlas Program (HuBMAP) research initiative. Your goal is to create a JSON elasticsearch query to search for objects that matches my requested search parameters. You should try to avoid using fields with free-form text values that are inconsistent from object to object. Make sure to include a match for "entity_type": "{entity_type}". Prefer using "match" or "range" instead of "term". Never use "filter" in your query, use "must" instead. None of the objects in the index are of "nested" type, so you must never use "nested" in your query. Use wildcard if there is a * star in the example values. If you need a field that is not described below, describe the field you are missing and do not return a query. Your query should use the minimum number of fields necessary to fulfill my search parameters. Do not use extra fields the user did not ask for. You do not need to use all of the fields below. Compress the json output removing spaces. Give me only the json code part of the answer. 

My search parameters: 
{question}

Fields:
{context}
"""

# Output format:

# ```json
# [your query, if you were able to construct one]
# ```
# [your explanation]

# None of the objects in the index are of "nested" type, so you must never use "nested" in your query.

# Output format:

# ```json
# [your query, if you were able to construct one]
# ```
# """

ES_SEARCH_FUNCTIONS = [
    # {
    #     "name": "submit_query", 
    #     "description": "Submit a query",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "A minified JSON elasticsearch query. DO NOT include whitespaces or newlines.",
    #             },
    #             "explanation": {
    #                 "type": "string",
    #                 "description": "Explanation of the query you constructed, the names of the fields you chose to use, and why. You can use markdown.",
    #             },
    #         },
    #         "required": ["query", "explanation"],
    #     },
    # },
    # {
    #     "name": "failed",
    #     "description": "Report a failed query",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "reason": {
    #                 "type": "string",
    #                 "description": "Apologize and explain why you could not create the query, and give suggestions for how the user can fix their query. You can use markdown.",
    #             },
    #         },
    #         "required": ["reason"],
    #     },
    # },
]

LIKELY_FOLLOWUP_PROMPT = """You are a chatbot on the website for the Human BioMolecular Atlas Program (HuBMAP) research initiative. You are capable of answering questions about the initiative, or running searches for HuBMAP donors, samples, or datasets. The following is the chat history between you and the user. Based on this chat history, come up with three distinct follow-up questions that the user is likely to ask that you are capable of answering.
  
Chat history:
{chat_history}
"""

LIKELY_FOLLOWUP_FUNCTIONS = [
    {
        "name": "submit_followup",
        "description": "Submit follow-up questions",
        "parameters": {
            "type": "object",
            "properties": {
                "followup_1": {
                    "type": "string",
                },
                "followup_2": {
                    "type": "string",
                },
                "followup_3": {
                    "type": "string",
                },
            },
            "required": ["followup_1", "followup_2", "followup_3"],
        },
    },
]

NO_RESULTS_FOUND_MSG = """I ran your query and found no results.
You can submit the same request again, and I will try a different query.
You can also take a look at the JSON query above and try rephrasing your request.
"""

EXPLAIN_PROMPT = """
Translate this query into natural language, starting with "Searching for..." and ending with ...: {query}
"""

EXPLAIN_JSON = """
Translate this query a list of code-like criteria quoted using `.
You can use markdown in your response. 
Your response should start with "Searching for {entity} objects that match the following criteria:"
"""

EXPLAIN_JSON2 = """
Create code-like summaries of elasticsearch json queries in the following format.
Example input: {"query": {"bool": {"must": [{"match": {"entity_type": "Dataset"}}, {"range": {"published_timestamp": {"gte": 1678370532036}}}, {"bool": {"must": [{"wildcard": {"files.rel_path": "*.tiff"}}, {"range": {"files.size": {"gt": 5000000000, "lt":1000000000}}}]}}]}}}

Example output:
Searching for objects that match the following criteria:
- `entity_type` matches "Dataset"
- `published_timestamp` >= 1678370532036
- `files.rel_path` matches "*.tiff"
- 1000000000 < `files.size` < 5000000000
"""

def hardcoded_explain(parsed_query):
    objs = []
    def traverse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["match", "range", "wildcard"]:
                    objs.append((key, value))
                else:
                    traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)
        else:
            pass

    traverse(parsed_query)

    expl = "Searching for objects that match the following criteria:\n"

    for k, obj in objs:
        if isinstance(obj, dict):
            for field, value in obj.items():
                term = value
                if k == "match":
                    if isinstance(term, dict):
                        term = term.get("query", term.get("value", None))
                    if term is not None and isinstance(term, str):
                        expl += f"- `{field}` matches \"{term}\"\n"
                elif k == "range":
                    if isinstance(term, dict):
                        gte = term.get("gte", None)
                        gt = term.get("gt", None)
                        lte = term.get("lte", None)
                        lt = term.get("lt", None)
                        if gte is not None:
                            expl += f"- `{field}` >= {gte}\n"
                        if gt is not None:
                            expl += f"- `{field}` > {gt}\n"
                        if lte is not None:
                            expl += f"- `{field}` <= {lte}\n"
                        if lt is not None:
                            expl += f"- `{field}` < {lt}\n"
                elif k == "wildcard":
                    if isinstance(term, dict):
                        term = term.get("value", term.get("wildcard", None))
                    if term is not None and isinstance(term, str):
                        escapedterm = markdown_strings.esc_format(term)
                        expl += f"- `{field}` matches \"{escapedterm}\"\n"
                    
    return expl

def _create_retry_decorator(max_retries) -> Callable[[Any], Any]:
    import openai

    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


retry_dec = _create_retry_decorator(max_retries=5)


@retry_dec
def retry_chat_completion(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)


@dataclass
class ParsedQuery:
    query: str



def condense_history(prev_user_message, prev_chatbot_response, user_message: str, model_name) -> str:

    if prev_user_message is None or prev_chatbot_response is None:
        return user_message

    prev_chatbot_response = "[some response here]"

    chat_history = f"User: {prev_user_message}\n\nAssistant: {prev_chatbot_response}"

    formatted_prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=chat_history,
        question=user_message
    )

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")

    choice = completion["choices"][0]
    condensed = choice["message"]["content"]

    logger.debug(f"condensed standalone: {condensed}")

    return condensed

class SelectException(Exception):
    def __init__(self, message):
        super().__init__(message)

def select_engine(user_message, model_name):
    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": RETRIEVER_SELECT_PROMPT.format(question=user_message)},
            # {"role": "system", "content": RETRIEVER_SELECT_PROMPT.format(question=user_message)},
            # {"role": "user", "content": user_message},
            {"role": "system", "content": "Do not respond to the user's question."},
        ],
        functions=RETRIEVER_SELECT_FUNCTIONS,
        function_call={"name": "categorize_question"},
        temperature=0,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")

    choice = completion["choices"][0]

    if "function_call" not in choice["message"]:
        content = choice["message"]["content"]
        logger.debug(f"select engine content: {content}")
        usr_message = "I couldn't understand your question. Please try rephrasing your question, or try asking a different question."
        # return ChatMessage(role=Roles.ASSISTANT, summary_content=usr_message, renderable_content=[usr_message])
        raise SelectException(usr_message)
        # return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=["didn't call a function", content])
    else:
        function_call = choice["message"]["function_call"]
        arg_dict = json.loads(function_call["arguments"])

        logger.debug(f"Selected function_call: {function_call}")
        logger.debug(f"Response arg_dict: {arg_dict}")

        category_arg = arg_dict["category"]
        # selected_engine = ENGINE_MAPPING[category_arg]
        # selected_engine = ENGINE_MAPPING[function_call["name"]]
        msg = f"category: {category_arg}"



        if "entity_type" in arg_dict:
            msg += f", entity_type: {arg_dict['entity_type']}"
        logger.debug(msg)
        # return ChatMessage(role=Roles.ASSISTANT, summary_content=msg, renderable_content=[msg])
        SEARCH_MAPPING = {
            "dataset": Modes.SEARCH_DATASET,
            "sample": Modes.SEARCH_SAMPLE,
            "donor": Modes.SEARCH_DONOR,
        }

        engine_name = category_arg

        if category_arg == "search":
            if "entity_type" not in arg_dict:
                # return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Missing entity_type", renderable_content=[f"Missing entity_type"])
                raise SelectException(f"Missing entity_type")
            if arg_dict["entity_type"] not in SEARCH_MAPPING:
                # return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown entity type: {arg_dict['entity_type']}", renderable_content=[f"Unknown entity type: {arg_dict['entity_type']}"])
                raise SelectException(f"Unknown entity type: {arg_dict['entity_type']}")
            engine_name = SEARCH_MAPPING[arg_dict["entity_type"]]

        if engine_name not in REDUCE_ENGINE_MAP:
            # return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown question category: {engine_name}", renderable_content=[f"Unknown question category: {engine_name}"])
            raise SelectException(f"Unknown question category: {engine_name}")
        
    logger.debug(f"selected engine: {engine_name}")
        
    return engine_name

def auto_engine(user_message, model_name):
    try:
        res = select_engine(user_message, model_name)
    except SelectException as e:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=str(e), renderable_content=[str(e)])
    engine_name = res
    reduced_engine = REDUCE_ENGINE_MAP[engine_name]
    if reduced_engine == Modes.AUTO:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown question category: {engine_name}", renderable_content=[f"Unknown question category: {engine_name}"])
    return ENGINE_MAPPING[reduced_engine](user_message, model_name)


def simple_engine(user_message: str, model_name) -> str:
    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": SIMPLE_ENGINE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")
    choice = completion["choices"][0]
    content = choice["message"]["content"]

    logger.debug(f"simple engine content: {content}")
    return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])


def general_engine(user_message: str, model_name) -> str:
    docs = RETRIEVERS[Modes.GENERAL].get_relevant_documents(user_message)

    doc_token_limit = 2500
    docs = reduce_tokens_below_limit(doc_token_limit, docs)

    logger.debug(f"docs:")
    for doc in docs:
        logger.debug(doc.page_content)

    formatted_prompt = TEXT_ANSWER_PROMPT.format(
        context="\n\n\n".join([doc.page_content for doc in docs]),
        question=user_message
    )

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        # functions=TEXT_ANSWER_FUNCTIONS,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")
    choice = completion["choices"][0]

    if choice["finish_reason"] != "function_call":
        content = choice["message"]["content"]
        # display_chatbot_message(content)
        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])
    else:
        content = TEXT_ANSWER_FAIL_MSG
        # display_chatbot_message(content)
        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])


def es_engine(user_message: str, model_name, entity_type=None, retry=1) -> str:
    # assert entity_type in ["dataset", "sample", "donor"]

    # entity_type = Modes.SEARCH_DATASET  # TODO: remove this line

    entity_formatted_mapping = {
        Modes.SEARCH_DATASET: "Dataset",
        Modes.SEARCH_SAMPLE: "Sample",
        Modes.SEARCH_DONOR: "Donor",
    }

    if entity_type not in entity_formatted_mapping:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown entity type: {entity_type}", renderable_content=[f"Unknown entity type: {entity_type}"])

    entity_type_formatted = entity_formatted_mapping[entity_type]
    extras = f"entity_type {entity_type_formatted}"

    logger.debug(f"searching with user_message: {user_message}")

    docs = RETRIEVERS[entity_type].get_relevant_documents(user_message+" "+extras)
    # docs = RETRIEVERS[f"{RetrieverOptions.SEARCH}_{entity_type}"].get_relevant_documents(user_message+" "+extras)

    if len(docs) == 0:
        raise Exception("No docs found")
    

    doc_token_limit = 4000
    # doc_token_limit = 2000
    docs = reduce_tokens_below_limit(doc_token_limit, docs)

    logger.debug(f"docs:")
    for doc in docs:
        logger.debug(doc.page_content)

    if not os.environ.get("DOCKER_CONTAINER", False):
        log_folder = Path(os.environ.get("LOG_DIR", "logs"))
        log_folder.mkdir(exist_ok=True, parents=True)
        with open(log_folder/"out2.log", "w") as f:
            f.write("\n\n\n".join([doc.page_content for doc in docs]))

    if len(docs) == 0:
        raise Exception("All docs were too long")
    
    # raise Exception("TODO")

    context = "\n\n\n".join([doc.page_content for doc in docs])
    context = context.replace("[n]", "")
    # context = context.replace("example values:", '(not a nested type, use "bool" "must" "match/range/wildcard") example values:')
    # context = context.replace("example values:", '(never use in "filter", use "must") example values:')

    formatted_prompt = ES_SEARCH_PROMPT.format(
        context=context,
        question=user_message,
        entity_type=entity_type_formatted,
    )

    MAX_TOKENS_IN_INPUT = 3600
    formatted_prompt = ENCODING.decode(ENCODING.encode(formatted_prompt)[:MAX_TOKENS_IN_INPUT])

    if not os.environ.get("DOCKER_CONTAINER", False):
        log_folder = Path(os.environ.get("LOG_DIR", "logs"))
        log_folder.mkdir(exist_ok=True, parents=True)
        with open(log_folder/"out3.log", "w") as f:
            f.write(formatted_prompt)

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        n=3,
        # functions=ES_SEARCH_FUNCTIONS,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")
    # global choice
    # choice = completion["choices"][0]

    results = [parse_es_choice(choice, entity_type_formatted) for choice in completion["choices"]]

    num_succeeded = sum([res.successful for res in results])
    logger.debug(f"num_succeeded: {num_succeeded}")

    if num_succeeded == 0 and retry > 0:
        results += es_engine(user_message, model_name, retry=retry-1)

    # first_success = next((res for res in results if res.successful), None)
    # if first_success:
    #     summary = first_success[1].summary_content
    # else:
    #     summary = results[0][1].summary_content

    # sort results, succeeded first
    results.sort(key=lambda res: res.num_results)
    results.sort(key=lambda res: res.successful, reverse=True)

    return results

def parse_es_choice(choice, entity_type_formatted):
    did_succeed = False

    if choice["finish_reason"] != "function_call":
        content = choice["message"]["content"]

        res = []

        found_json_str = re.search(r"```json(.*?)```", content, re.DOTALL)
        if not found_json_str:
            found_json_str = re.search(r"```(.*?)```", content, re.DOTALL)

        if found_json_str:
            everything_besides_json = re.sub(r"```json(.*?)```", "", content, flags=re.DOTALL)
            everything_besides_json = everything_besides_json.strip()

            res.append(everything_besides_json)

            json_query = found_json_str.group(1).strip()
            logger.debug(f"json_query: {json_query}")
        else:
            json_query = content.strip()
            logger.debug(f"json_query: {json_query}")

        if json_query is None:
            logger.debug("No parsable json query found")
            return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed)

        try:
            parsed_query = json.loads(json_query)
        except json.decoder.JSONDecodeError as e:
            logger.debug(f"JSONDecodeError: {e}")
            msg = "Sorry, I wasn't able to create a valid JSON query. Please try rephrasing your question, or try asking a different question."
            res.append(e)
            res.append(msg)
            return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed)
        
        if "query" not in parsed_query:
            parsed_query = {"query": parsed_query}
            logger.debug("No query key found in parsed query, adding one")

        logger.debug(parsed_query)
        # res.append("JSON query:")
        res.append(ParsedQuery(query=parsed_query))

        num_results = 0

        try:
            # global search_results, formatted_search_results
            search_results = hubmap_search(parsed_query)
            formatted_search_results = format_search_result_to_dataframe(search_results) # dataframe
            logger.debug(f"search_results: {search_results}")

            res.append(hardcoded_explain(parsed_query))

            # completion = retry_chat_completion(
            #     messages=[
            #         {"role": "system", "content": EXPLAIN_JSON2},
            #         {"role": "user", "content": json.dumps(parsed_query)},
            #     ],
            #     n=3,
            #     # functions=ES_SEARCH_FUNCTIONS,
            #     **CHAT_COMPLETION_DEFAULT_KWARGS
            # )

            # choice = completion["choices"][0]
            # content = choice["message"]["content"]
            # # replace /n with /n/n
            # content = re.sub(r"\n", r"\n\n", content)
            # res.append(content)

            num_results = len(formatted_search_results)
            if num_results > 0:
                formatted_results_msg = f"I ran this query and found {num_results} results:"
                res.append(formatted_results_msg)
                res.append(formatted_search_results)
                did_succeed = True
            else:
                res.append(NO_RESULTS_FOUND_MSG)
        except HTTPException as e:
            logger.debug(f"HuBMAP es search failed: {e}")
            msg = "Sorry, I wasn't able to complete your search. Please try rephrasing your question, or try asking a different question."
            res.append(msg)
            res.append(e)


        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed, num_results=num_results)

    else:
        function_call = choice["message"]["function_call"]
        arguments = json.loads(function_call["arguments"])
        if function_call.name == "failed":
            display_chatbot_message(content)
            # return [arguments["reason"]]
            res = arguments["reason"]
            return ChatMessage(role=Roles.ASSISTANT, summary_content=res, renderable_content=[res], successful=did_succeed)
        else:
            # display_chatbot_message(content)
            return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content], successful=did_succeed)
        # elif function_call.name == "submit_query":
        #     json_query = arguments["query"]
        #     md_json_query = JSON_MD.format(json=json.dumps(json.loads(json_query), indent=2))
        #     explanation = arguments["explanation"]
        #     return f"{explanation}\n\nHere is the query:\n\n{md_json_query}"

ALWAYS_INCLUDES = [
            # "url",
            "uuid",
            "hubmap_id",
            # "donor.mapped_metadata.death_event",
            # "donor.mapped_metadata.mechanism_of_injury",
            # "donor.mapped_metadata.sex",
            # "donor.mapped_metadata.age_value",
            # "donor.mapped_metadata.race",
            # "provider_info",
            # "files.type",
            # "source_samples.created_by_user_email",
            # "donor.mapped_metadata.medical_history"
        ]

REQUEST_EXTRAS = {
    "_source": {
        "excludes": [
            "ancestors",
            "descendants",
            "immediate_ancestors",
            "immediate_descendants",
            "metadata",
            "donor.metadata"
        ],
    },
    "size": 10000,
}


def hubmap_search(request):
    full_request = copy.deepcopy(request)
    for key, value in REQUEST_EXTRAS.items():
        full_request[key] = value

    full_request["_source"]["includes"] = list(set(ALWAYS_INCLUDES + get_checked_fields(full_request)))
    print(full_request["_source"]["includes"])
    search_result = SEARCHSDK_INSTANCE.search_by_index(full_request, "portal")
    return search_result

def get_checked_fields(request):
    query_body = request["query"]
    # recusive find any objects with key "match", "range", or "wildcard"

    checked_fields = []

    def get_checked_fields_recursive(query_body):
        if isinstance(query_body, dict):
            for key in query_body:
                if key in ["match", "range", "wildcard"]:
                    checked_fields.append(list(query_body[key].keys())[0])
                else:
                    get_checked_fields_recursive(query_body[key])
        elif isinstance(query_body, list):
            for item in query_body:
                get_checked_fields_recursive(item)

    get_checked_fields_recursive(query_body)


    return checked_fields


def get_field_example_dict(json_obj, field_example_dict, prefix=""):
    if isinstance(json_obj, dict):
        for key in json_obj:
            if prefix == "":
                new_prefix = key
            else:
                new_prefix = prefix + "." + key
            get_field_example_dict(json_obj[key], field_example_dict, new_prefix)
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            new_prefix = prefix + f"[{i}]"
            get_field_example_dict(item, field_example_dict, new_prefix)
    else:
        field_example_dict[prefix] = json_obj
        # field_example_dict[prefix].add(json_obj)
        # field_example_dict[prefix][json_obj] += 1


def format_search_result_to_dataframe(search_result):
    hits = search_result["hits"]["hits"]

    hits_firstval = []
    for hit in hits:
        val_dict = defaultdict(None)
        get_field_example_dict(hit["_source"], val_dict)
        hits_firstval.append(val_dict)

    df = pd.DataFrame(hits_firstval)

    if "uuid" in df.columns:
        df["url"] = df["uuid"].apply(lambda x: f"https://portal.hubmapconsortium.org/browse/dataset/{x}")

    columns = list(df.columns)
    columns.sort()
    if "url" in columns:
        columns.remove("url")
        columns.insert(0, "url")
    df = df[columns]
    return df


@lru_cache(maxsize=1)
def is_running_in_notebook():
    try:
        get_ipython().config
        return True
    except NameError:
        return False


SEARCH_SAMPLE_REQUEST = {
    "query": {
        "bool": {
            "must": [
                {"match": {"provider_info": "Stanford"}},
                {"match": {"entity_type": "Dataset"}}
            ]
        }
    },
    "size": 5
}

PYTHON_SEARCH_SAMPLE_TEMPLATE = """
# !pip install hubmap-sdk
import json
from hubmap_sdk import SearchSdk

service_url = "{service_url}"
searchsdk_instance = SearchSdk(service_url=service_url)

# for more info on elasticsearch query syntax, see https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
search_json = \"\"\"
{search_json}
\"\"\"

search_result = searchsdk_instance.search_by_index(json.loads(search_json), "portal")

print(json.dumps(search_result, indent=2))
"""


def python_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict, indent=2)
    return PYTHON_SEARCH_SAMPLE_TEMPLATE.format(search_json=search_json, service_url=service_url)


CURL_SEARCH_SAMPLE_TEMPLATE = """
curl -X POST "{service_url}search" -H "accept: application/json" -H "Content-Type: application/json" -d {search_json}
"""


def curl_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict)
    return CURL_SEARCH_SAMPLE_TEMPLATE.format(search_json=shlex.quote(search_json), service_url=service_url)

R_SEARCH_SAMPLE_TEMPLATE = """
library(httr)
library(jsonlite)

service_url <- "{service_url}"

search_json <- '{search_json}'

response <- POST(paste0(service_url, "search"), body = search_json, encode = "json", verbose())

content(response, "text")
"""

def r_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict)
    return R_SEARCH_SAMPLE_TEMPLATE.format(search_json=search_json, service_url=service_url)

EMBEDDINGS = OpenAIEmbeddings(openai_api_key="placeholder")

VECTORSTORES = {}
for vectorstore_name, vectorstore_info in VECTORSTORE_INFOS.items():
    print(f"loading database {vectorstore_name} from {vectorstore_info.path} with collection name {vectorstore_info.col}")

    vectorstore = Chroma(persist_directory=str(vectorstore_info.path), embedding_function=EMBEDDINGS, collection_name=vectorstore_info.col)
    VECTORSTORES[vectorstore_name] = vectorstore

RETRIEVERS = {
    Modes.GENERAL: VECTORSTORES["website"].as_retriever(search_kwargs=kwargs_to_dict(k=20)),
    Modes.SEARCH_DATASET: VECTORSTORES["es_dataset"].as_retriever(search_kwargs=kwargs_to_dict(k=100)),
}

ENGINE_MAPPING = {
    Modes.AUTO: auto_engine,
    Modes.GENERAL: general_engine,
    Modes.SEARCH_DATASET: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET),
    Modes.SEARCH_SAMPLE: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET),
    Modes.SEARCH_DONOR: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET),
    Modes.OTHER: simple_engine,
}

REDUCE_ENGINE_MAP = {
    Modes.AUTO: Modes.AUTO,
    Modes.GENERAL: Modes.GENERAL,
    Modes.SDK: Modes.GENERAL,
    Modes.INGEST: Modes.GENERAL,
    Modes.SEARCH_DATASET: Modes.SEARCH_DATASET,
    Modes.SEARCH_SAMPLE: Modes.SEARCH_DATASET,
    Modes.SEARCH_DONOR: Modes.SEARCH_DATASET,
    Modes.OTHER: Modes.OTHER,
    "about": Modes.GENERAL,
    "about_search": Modes.GENERAL,
}



def display_chatbot_message(message):
    display_markdown(f'#### Chatbot:\n{message}')


def display_user_message(message):
    display_markdown(f'#### User:\n{message}')


def display_markdown(markdown):
    is_jupyter_notebook = is_running_in_notebook()
    if is_jupyter_notebook:
        from IPython.display import display, Markdown
        display(Markdown(markdown))
    else:
        console = rich.console.Console()
        md = rich.markdown.Markdown(markdown)
        console.print(md)

def main():
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(filename='simple_chatbot.log', filemode='w', level=logging.DEBUG)

    prev_user_message = None
    prev_chatbot_response = None

    display_chatbot_message(INITIAL_MESSAGE)
    while True:

        user_message = input("User: ")
        if len(user_message) == 0:
            display_chatbot_message("Goodbye!")
            break
        display_user_message(user_message)
        user_message = condense_history(prev_user_message, prev_chatbot_response, user_message)
        logger.debug(f"Standalone user question: {user_message}")

        chatbot_response = general_engine(user_message)

        for renderable_content in chatbot_response.renderable_content:
            display_chatbot_message(renderable_content)

        prev_user_message = user_message
        prev_chatbot_response = chatbot_response

if __name__ == "__main__":
    main()

