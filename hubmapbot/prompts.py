from collections import defaultdict
from constants import *

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