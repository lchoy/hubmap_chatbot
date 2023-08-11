import os
import time
from typing import Union
from pandas import DataFrame

import streamlit as st

from hubmapbot.simple_chatbot import *
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')


handlers = [logging.StreamHandler()]
if not os.environ.get("DOCKER_CONTAINER", False):
    log_folder = Path(os.environ.get("LOG_DIR", "logs"))
    log_folder.mkdir(exist_ok=True, parents=True)
    handlers.append(logging.FileHandler(log_folder/'app.log'))
    
logging.basicConfig(level=logging.DEBUG, handlers=handlers)

st.set_page_config(layout="wide")
st.title("HuBMAP Chatbot")


def reset_chat_history():
    try:
        del st.session_state.messages
    except:
        pass

def validate_api_key(key):
    return key is not None and len(key) > 0 and key.startswith("sk-")

def validate_api_org(org):
    return org is not None and (len(org) == 0 or org.startswith("org-"))

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_ORGANIZATION_ENV = "OPENAI_ORGANIZATION"
BAD_API_KEY = "bad_api_key"

def set_api_key(key, org):
    openai.api_key = key
    openai.organization = org
    EMBEDDINGS.openai_api_key = key
    st.session_state[OPENAI_API_KEY_ENV] = key
    st.session_state[OPENAI_ORGANIZATION_ENV] = org
    st.session_state[BAD_API_KEY] = False
    st.session_state["last_exception"] = None

def reset_api_key():
    environ_api_key = os.getenv(OPENAI_API_KEY_ENV, "")
    environ_api_org = os.getenv(OPENAI_ORGANIZATION_ENV, "")
    set_api_key(environ_api_key, environ_api_org)
    st.session_state['bad_api_key'] = True

with st.sidebar:
    st.header("HuBMAP Chatbot")
    st.button("Reset API Key 🔑", on_click=reset_api_key, use_container_width=True)
    st.button("Clear chat history 🧹", on_click=reset_chat_history, use_container_width=True)
    mode_str = st.selectbox("Select a mode:", list(MODE_MAP.keys()), key="mode_select")
    mode_select = MODE_MAP[mode_str]

    model_str = st.selectbox("Select a model:", list(MODEL_MAP.keys()), key="model_select")
    model_name = MODEL_MAP[model_str]



openai_api_key = st.session_state.get(OPENAI_API_KEY_ENV, os.getenv(OPENAI_API_KEY_ENV, ""))
openai_api_org = st.session_state.get(OPENAI_ORGANIZATION_ENV, os.getenv(OPENAI_ORGANIZATION_ENV, ""))

valid_api_key = validate_api_key(openai_api_key)
valid_api_org = validate_api_org(openai_api_org)

bad_key_override = st.session_state.get(BAD_API_KEY, False)

if OPENAI_API_KEY_ENV not in st.session_state:
    set_api_key(openai_api_key, openai_api_org)

if not valid_api_key or not valid_api_org or bad_key_override:
    if st.session_state.get("last_exception", None) is not None:
        st.error(st.session_state["last_exception"])
    new_openai_api_key = st.text_input("OpenAI API Key [(create one here)](https://platform.openai.com/account/api-keys)", type="password", placeholder="e.g. sk-abc123lH9nREzdMHNT9J1Ex1x2o8frzKzyUOdyH9HmH", value=openai_api_key)
    new_openai_api_org = st.text_input("OpenAI Organization ID (Leave blank to use your default organization) [(find yours here)](https://platform.openai.com/account/org-settings)", type="password", placeholder="e.g. org-abc123qiDp3ZWhOZchF6Pf", value=openai_api_org)
    submit = st.button("Submit API key", use_container_width=True)
    if submit:
        valid_api_key = validate_api_key(new_openai_api_key)
        valid_api_org = validate_api_org(new_openai_api_org)
        if valid_api_key and valid_api_org:
            set_api_key(new_openai_api_key, new_openai_api_org)
            st.info("API key set successfully.")
            st.experimental_rerun()
        else:
            if not valid_api_key:
                st.error("Invalid API key.")
            if not valid_api_org:
                st.error("Invalid organization ID.")
    st.stop()

assistant_avatar = "hubmap_user.jpg"
avatar_dict = {
    Roles.ASSISTANT: assistant_avatar,
}


def add_new_prompt(prompt, role=Roles.USER):
    # if prompt is not string, raise error
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    message_obj = ChatMessage(role=role, summary_content=prompt, renderable_content=[prompt])
    st.session_state.messages.append(message_obj)
    return message_obj


def render_multi_msg(msg: Union[ChatMessage, List[ChatMessage]], chat_msg=None):
    msglist = None
    if isinstance(msg, list):
        msglist = msg
        msg = msglist[0]

    role = msg.role
    
    if chat_msg is None:
        with chat_container:
            chat_msg = st.chat_message(role, avatar=avatar_dict.get(role, None))


    with chat_msg:
        if msglist is None:
            draw_message_contents(msg)
        else:
            res_str_arr = [" (failed)" if not msg.successful else "" for msg in msglist]
            names = [f"Response {i+1}{res_str_arr[i]}" for i in range(len(msglist))]
            tabs = st.tabs(names)
            for i, msg in enumerate(msglist):
                with tabs[i]:
                    draw_message_contents(msg)


def draw_message_contents(msg):

    content = msg.renderable_content

    assert isinstance(content, list)

    for content_item in content:
        if isinstance(content_item, ParsedQuery):

            with st.expander("Show request details"):
                tabmap = {
                    "JSON": lambda: st.json(content_item.query),
                    "Python": lambda: st.code(python_example_search(content_item.query), language="python"),
                    "cURL": lambda: st.code(curl_example_search(content_item.query), language="bash"),
                    "R": lambda: st.code(r_example_search(content_item.query), language="r"),
                    "raw": lambda: st.code(json.dumps(content_item.query), language="json"),
                }
                tablist = list(tabmap.keys())
                tabs = st.tabs(tablist)
                for i, tabname in enumerate(tablist):
                    with tabs[i]:
                        tabmap[tabname]()

        elif isinstance(content_item, HTTPException):
            with st.expander("Error details"):
                st.write(content_item)
                parsed_e_body = content_item.get_description()
                try:
                    parsed_e_body = json.loads(content_item.get_description())
                except:
                    pass
                if parsed_e_body:
                    st.write(parsed_e_body)

        elif isinstance(content_item, DataFrame):
            column_config = {        
                "url": st.column_config.LinkColumn(
                    "url",
                    width="small",
                )
            }
            st.dataframe(content_item, column_config=column_config)
        else:
            st.write(content_item)

if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    reset_chat_history()
    st.session_state.messages = []
    add_new_prompt(INITIAL_MESSAGE, role=Roles.ASSISTANT)

chat_container = st.container()

for msg in st.session_state.messages:
    render_multi_msg(msg)

if prompt := st.chat_input(placeholder="Type a question...", key="chat_input"):
    stripped_prompt = prompt.strip()
    if stripped_prompt.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = stripped_prompt
        reset_chat_history()
        st.experimental_rerun()
    render_multi_msg(add_new_prompt(prompt))

suggestions_parent = st.empty()
suggestions_parent.write()


def get_last_msgs():
    prev_user_content = None
    prev_chatbot_content = None
    try:
        last_usr_msg = st.session_state.messages[-3]
        last_chatbot_msg = st.session_state.messages[-2]

        assert last_usr_msg.role == Roles.USER
        assert last_chatbot_msg.role == Roles.ASSISTANT

        prev_user_content = last_usr_msg.summary_content
        prev_chatbot_content = last_chatbot_msg.summary_content
    except:
        pass
    return prev_user_content, prev_chatbot_content


def self_or_first(x):
    if isinstance(x, list):
        return x[0]
    return x

if self_or_first(st.session_state.messages[-1]).role == Roles.USER:
    user_message = st.session_state.messages[-1].summary_content

    with chat_container:
        chat_msg = st.chat_message("assistant", avatar=avatar_dict.get(Roles.ASSISTANT, None))

    if not openai_api_key:
        response_obj = ChatMessage(role=Roles.ASSISTANT, summary_content=INVALID_API_KEY_MSG, renderable_content=[INVALID_API_KEY_MSG])
    else:

        with chat_msg:
            with st.spinner('Thinking...'):

                prev_user_message, prev_chatbot_response = get_last_msgs()

                logger.info(f"User message: {user_message}")
                try:
                    chatbot_response = ENGINE_MAPPING[REDUCE_ENGINE_MAP[mode_select]](user_message, model_name)
                except openai.error.AuthenticationError as e:
                    st.session_state['bad_api_key'] = True
                    st.session_state['last_exception'] = e
                    st.experimental_rerun()
                except Exception as e:
                    logger.error(e)
                    chatbot_response = ChatMessage(role=Roles.ASSISTANT, summary_content="An error occurred.", renderable_content=["An error occurred:", e])

                response_obj = chatbot_response

    st.session_state.messages.append(response_obj)
    render_multi_msg(response_obj, chat_msg=chat_msg)

with suggestions_parent:
    suggestions_container = st.container()

with suggestions_container:
    st.markdown(f"<p style='text-align: center;'><b>Mode:</b> {mode_str}</p>", unsafe_allow_html=True)

    for a in INITIAL_SUGGESTIONS_MAP[mode_select]:
        st.button(a, on_click=lambda a=a: add_new_prompt(a))
