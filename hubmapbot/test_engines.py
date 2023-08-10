# pytest -k test_select -s --log-cli-level INFO

from dotenv import load_dotenv
import logging


from collections import defaultdict
from .simple_chatbot import *
import os


load_dotenv(dotenv_path='.env')

logger = logging.getLogger(__name__)

handlers = [logging.StreamHandler()]
if not os.environ.get("DOCKER_CONTAINER", False):
    log_folder = Path(os.environ.get("LOG_DIR", "logs"))
    log_folder.mkdir(exist_ok=True, parents=True)
    handlers.append(logging.FileHandler(str(log_folder/'engine_tests.log')))
logging.basicConfig(level=logging.DEBUG, handlers=handlers)

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is not None and len(openai_api_key) > 0:
    openai.api_key = openai_api_key
    EMBEDDINGS.openai_api_key = openai_api_key
    # st.session_state['openai_api_key'] = openai_api_key
    logger.debug("OpenAI API key set successfully.")
else:
    logger.debug("OpenAI API key not set.")



TEST_PROMPTS = defaultdict(list)
TEST_PROMPTS[Modes.GENERAL] = [
    # "What is HuBMAP?",
    # "What is the HuBMAP Data Portal?",
    # "What are HuBMAP IDs?",
    # "How do I download bulk data from hubmap?",
    # "Does HuBMAP provide Millitomes?",
]
TEST_PROMPTS[Modes.SEARCH_DATASET] = [
    # "Search for datasets are derived from male donors with diabetes",
    # "Find datasets derived from white female donors who died from natural causes",
    # "Find datasets containing arrow files",
    # "Find datasets published after 2020",
    # "Find datasets from donors who smoke or have smoked with a BMI less than 40",
    # "Find datasets from donors with type A blood",
    # "Find datasets from african-american donors between 45-72",
    "Search for spleen datasets published after 2021 with tiff files larger than 5 gigabytes",
    "Find CODEX [Cytokit + SPRM] datasets from donors with a weight between 80-100 kg and a height between 140-180cm",
    "Find datasets from white males between 25-35 years old with a history of cardiac arrest and hypertension who passed from anoxia",
    "Find left kidney datasets with a donor profile index under 50, where the donor is under 30 years old and is blood type O, and died from anoxia",
]


TEST_PROMPTS[Modes.AUTO] = {
    "How do I filter datasets by creation date?": "general",
    "What does the 'name' field in donor objects mean?": "general",
    "Perform a search.": "general",
    "Search for datasets.": "general",
    "Search for donors.": "general",
}

for prompt in TEST_PROMPTS[Modes.SEARCH_DATASET]:
    TEST_PROMPTS[Modes.AUTO][prompt] = "search_dataset"

for prompt in TEST_PROMPTS[Modes.GENERAL]:
    TEST_PROMPTS[Modes.AUTO][prompt] = "general"

# MODEL_NAME = "gpt-4-0613"
MODEL_NAME = "gpt-3.5-turbo-0613"


def test_search_dataset():
    single_succeeded = []
    single_failed = []

    succeeded = []
    failed = []
    mode = Modes.SEARCH_DATASET
    for prompt in TEST_PROMPTS[mode]:
        logger.debug(f"Prompt: {prompt}")
        # chatbot_response = ChatMessage(role=Roles.ASSISTANT, summary_content="hello", renderable_content=["wow"], successful=False)
        chatbot_response = ENGINE_MAPPING[mode](prompt, MODEL_NAME)

        if isinstance(chatbot_response, list):
            for response in chatbot_response:
                logger.debug(f"Response: {response.summary_content}")
                if response.successful:
                    succeeded.append((prompt, response))
                else:
                    failed.append((prompt, response))

            chatbot_response = chatbot_response[0]
        logger.debug(f"Response: {chatbot_response.summary_content}")
        for message in chatbot_response.renderable_content:
            logger.debug(message)
        if chatbot_response.successful:
            single_succeeded.append((prompt, chatbot_response))
        else:
            single_failed.append((prompt, chatbot_response))

    summarize_results(single_succeeded, single_failed, succeeded, failed)


def test_select():
    single_succeeded = []
    single_failed = []
    
    succeeded = []
    failed = []
    for prompt, desired_mode in TEST_PROMPTS[Modes.AUTO].items():
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Mode: {desired_mode}")
        try:
            output_mode = select_engine(prompt, MODEL_NAME)
            reduced_mode = REDUCE_ENGINE_MAP[output_mode]
            logger.info(f"Category: {output_mode} ({reduced_mode})")
            if reduced_mode == desired_mode:
                succeeded.append((prompt, reduced_mode))
                single_succeeded.append((prompt, reduced_mode))
            else:
                logger.warn(f"Failed: {prompt} ({reduced_mode} != {desired_mode})")
                failed.append((prompt, reduced_mode))
                single_failed.append((prompt, reduced_mode))
        except SelectException as e:
            logger.warn(f"Failed: {prompt} ({desired_mode}) ({e})")
            failed.append((prompt, None))
            single_failed.append((prompt, None))

    summarize_results(single_succeeded, single_failed, succeeded, failed)

        #     if len(single_failed) > 0:
        #         break
        # if len(single_failed) > 0:
        #     break

def summarize_results(single_succeeded, single_failed, succeeded, failed):
    single_total = len(single_succeeded) + len(single_failed)
    single_num_succeeded = len(single_succeeded)
    single_num_failed = len(single_failed)

    total = len(succeeded) + len(failed)
    num_succeeded = len(succeeded)
    num_failed = len(failed)

    logger.info(f"=== SUMMARY ===")
    logger.info(f"Any: {single_num_succeeded} out of {single_total} tests succeeded.")
    logger.info(f"Total: {num_succeeded} out of {total} tests succeeded.")

    logger.info(f"=== FAILED TESTS: ===")
    for prompt, response in single_failed:
        logger.info(f"Prompt: {prompt}")
        if response is not None:
            logger.info(f"Response: {response}")



