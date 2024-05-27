__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from functions import *


openai_api_key = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_avatar1(role):
    avatars = {
        'user': '🥷',
        'assistant': '✨',
        'system': '⚙️'
    }
    return avatars[role]

st.set_page_config(
    page_title='Identity Security Conference Buddy',
    page_icon='✨',
    layout='wide',
    initial_sidebar_state='expanded',
)
#st.title("Identity Security Conference Buddy ")
with st.sidebar.form("ustr_url"):

    ustr_url = st.text_area("Knowledge Base Url Link - one per line")   
    ustr_submitted = st.form_submit_button("Get Content")


if os.path.exists("./kdb"):
    embedded = load_existing_kb()

if ustr_submitted:
    if ustr_url:
        ustr_urls = ustr_url.split("\n")
        embedded = load_new_kb(ustr_urls)
    if embedded:
        st.toast(
            f"Knowledge base updated successfully.", icon='🎉')
 
#print("######### Called again  #################")

    
# Streamlit

# --------------------- App --------------------- #
#st.set_page_config(page_title="Comprehend That", page_icon='✨', layout="wide", initial_sidebar_state="collapsed")
# Initialize session state
if 'prev_selected_prompt' not in st.session_state:
    #st.session_state.prev_selected_prompt = ""
    st.session_state.prev_selected_prompt = "What does LastMile do. Provide details about their IGA products"
# Top Row A
a2, a3 = st.columns(2)
#a1.markdown('..')
#a2.markdown('## ✨ Let Me Comprehend that ')
a2.markdown('## ✨ Hello again.  ')
#a2.markdown(' ')
a2.markdown(' ###### Your AI aide for Identity Security Conference (Gartner Security & Risk Management Summit) topics. ✨🥳🔒')
a2.markdown(' ')
selected_prompt = a2.selectbox('start with some faq or use custom prompt at the bottom.', ("What does LastMile do. Provide details about their IGA products", "I am a CISO attending Identiverse. What sessions should I attend.", "I am a CIAM administrator attending Identiverse. What sessions should I attend.", "I focus on authorization standards. Which sessions at Identiverse should I attend.", "List authentication and authorization frameworks and standards and provide details for each of them",  "Provide details of Tenets of zero trust security.", "Provide me a summary and takeaways from 'Introduction to Customer Identity and Access Management'.", "Provide me a bulleted summary of Identity proofing and its challenges.", "What is the Business case for IAM, provide a bulleted summary.", "What technical and business skill set are needed for Identiverse conference.", "What is zero trust architecture, provide details. List Identiverse sessions with topic as zero trust.", "Explain me CIAM concepts. List Identiverse sessions covering CIAM."))
a2.markdown(' ')
#custom_prompt=a2.chat_input(lang_data["ask_a_question"])
a3.image('img/faq2.png')

# Custom CSS - make avatars a bit bigger
custom_css = """
<style>
.stChatMessage div:first-child {
    font-size: 1.6em;
    border: none;
    background-color: inherit;
}
.embeddedAppMetaInfoBar_container__DxxL1 {display: none !important;}
.embeddedAppMetaInfoBar_hostedName__-kdmi {display: none !important;}
.embeddedAppMetaInfoBar_linkOutText__4ptMa {display: none !important;}
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
stHeader {visibility: hidden;}
.sidebar-container {margin-left: 0;}
div[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {
    display: none
}
[data-testid="stHeader"] {
    display: none
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html = True)

systemPrompt = "You are a friendly Identity Management Expert. \
You receive information and instructions from an Identiverse conference sessions and Knowledge websites. \
Use these to answer the questions as detailed and truthfully as possible. \
Do not provide general explanations, but only use the information and instructions from the Knowledge websites for your answers. \
If the answer is not included, answer 'I don't know 🤔.' and suggest similar questions from Knowledge websites that you can answer. \
Sprinkle a '✨ 🎉 🥳 🌟 👊 💥 💭 🔥 🚀 ' into your answers from time to time."


if True :
    if "messages" not in st.session_state:
        #st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state["messages"] = [{"role": "system", "content": systemPrompt}]
        st.session_state["messages"] = [{"role": "assistant", "avatar": "✨", "content": "How can I help you?"}]
    for msg in st.session_state.messages:
        #st.chat_message(msg["role"]).write(msg["content"])
        st.chat_message(msg["role"], avatar = get_avatar1(msg["role"])).markdown(msg["content"])
    
    custom_prompt=st.chat_input("How can I help you ?")
    # Determine the prompt to use based on selection change
    if st.session_state.prev_selected_prompt != selected_prompt:
        prompt = selected_prompt
    else:
        prompt = custom_prompt
    # Store the current selected prompt for future comparison
    st.session_state.prev_selected_prompt = selected_prompt

    print("######### User Query:   ", prompt)    
    if prompt :
    #= st.chat_input("How can I help you ?"):
        # Add prompt
        #st.session_state.messages.append({"role": "user", "content": prompt})
        #st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "avatar": "🥷", "content": prompt})
        st.chat_message("user", avatar = "🥷").write(prompt)


        with st.spinner(text="Asking LastMile.id ..."):
            #result  = answer_webUrls(prompt)
            result, sources  = retrieval_qa_inference(prompt)
            sources = ""
            full_response = ""
            message_placeholder = st.empty()

            #feedback = ""
            # Add answer and sources
            st.chat_message("assistant", avatar = "✨").write(result)
            st.session_state.messages.append({"role": "assistant", "avatar": "✨", "content": result})
            
            #with st.expander("Reference links ⚡📚", expanded=False):
            #    if 'metadata' in result:
            #        metadata = result['metadata']
            #        print(metadata)
            #        st.markdown(metadata)   
            #    st.markdown("* " + "\n* ".join("12"))
        
else:
    st.write("Please load URLs first.")
