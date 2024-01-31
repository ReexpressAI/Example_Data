"""
Demo browser-based streamlit UI for generating Reexpression Attributes from the Mixtral-8x7B-Instruct-v0.1 model,
which can then be copy-and-pasted directly into Reexpress for a corresponding model you have trained.

You can also use this to examine and debug the chat behavior of the model before running the command-line batch script
mixtral_batch_classification.py. Remember to be consistent with the input format tags [INST] and [/INST].

This assumes that the model is present in a folder named 'mlx_model_quantized' in this directory. We recommend
128 gb of RAM and an M2 Ultra or higher processor. This script has been tested with a
Mac Studio with a 76-core GPU with 128 GB of unified memory.

This script is intended as a simple demo, with input formatted for Mixtral-8x7B-Instruct-v0.1,
rather than a general purpose utility for all MLX LLMs.

This is based on https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/ and the example inference script at
https://github.com/ml-explore/mlx-examples/tree/main/llms/mixtral.

Run as via `streamlit run demo.py`, which will make the UI accessible from localhost and a corresponding URL on your
local network. To adjust the streamlit settings, please refer to the documentation at https://streamlit.io/.

Copyright © 2024 Reexpress AI, Inc.

"""

import streamlit as st
import os
import mlx.core as mx
import mixtral_batch_classification

max_allowed_tokens = 5000
kReaskMarker = "Re-ask"

st.set_page_config(page_title="Reexpression Attributes Generator (for Re-ask verification with Mixtral-8x7B-Instruct-v0.1)",
                   page_icon="🔵",
                   menu_items={
                       'Get Help': 'https://re.express/guide.html',
                       'Report a bug': "https://re.express/guide.html",
                       'About': "Demo/Tutorial companion script for use with **Reexpress one** (macOS application). **Reexpress one** is available today on the Mac App store. See https://re.express for more! Copyright 2024 Reexpress AI, Inc. All rights reserved. The MLX code is provided by Apple Inc. with an MIT License, and the Mixtral-8x7B-Instruct-v0.1 model weights are provided by Mistral AI with an Apache 2.0 license."
                   }
                   )

# Init total tokens. We keep track of total tokens in order to preempt inference over inputs that are excessively long.
# Currently, we cap at a length less than the models max possible length.
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 10


def is_reask_mode(generation_mode):
    return generation_mode == kReaskMarker


with st.sidebar:
    st.title(':blue[Reexpression Attributes] Generator')
    st.header("(Binary re-ask verifier)", divider="grey")
    generation_mode = st.radio(
        "Mode",
        [kReaskMarker, "Chat"],
        captions=["Generate attributes for binary Yes/No re-ask (for upload to Reexpress)",
                  "Standard chat for debugging"])

    temperature = 0.0
    if generation_mode == kReaskMarker:
        max_length = 1
    else:
        max_length = st.sidebar.slider('max_length', min_value=1, max_value=1000, value=200, step=10)

    st.subheader("Re-ask Template", help="Optional. Only used in Re-ask mode."
                                         " These are only for convenience to avoid retyping constants."
                                         " Be consistent with what was used in training the model in Reexpress one.",
                 divider="grey")
    if not is_reask_mode(generation_mode):
        st.info("Change mode to Re-ask to modify the template.")
    st.text_area(
        "Prompt text",
        key='prompt_text',
        help="Added before the document. E.g., 'Please classify the sentiment of the following review. Review:'",
        disabled=not is_reask_mode(generation_mode)
        )
    st.text_area(
        "Trailing instruction",
        key='trailing_instruction'
        "", help="Added after the 'Prompt text' and document and before 'Yes or No?'. "
                 "E.g., 'Question: Does the previous document have a positive sentiment?'",
        disabled=not is_reask_mode(generation_mode)
        )
    if is_reask_mode(generation_mode):
        st.subheader(":green[Template]", help="Template text is in green, if provided. 'DOCUMENT' is the text you provide "
                                              "in the chat interface in the main panel (right). In this version, "
                                              "'Yes or No?' is always appended to the end.")
        if len(st.session_state.trailing_instruction) > 0:
            st.markdown(f":green[{st.session_state.prompt_text}] DOCUMENT " + f":green[{st.session_state.trailing_instruction}] Yes or No?")
        else:
            st.markdown(f":green[{st.session_state.prompt_text}] DOCUMENT Yes or No?")

    def prefill_template_with_demo():
        st.session_state.prompt_text = "Please classify the sentiment of the following review. Review:"
        st.session_state.trailing_instruction = "Question: Does the previous document have a positive sentiment?"


    if is_reask_mode(generation_mode):
        st.button("Prefill Re-ask Template with tutorial example",
                  help="These are the options used in the corresponding tutorial in the Reexpress documentation."
                       " Modify accordingly for your task, noting that 'Yes or No?'"
                       " will always be appended to the end.",
                  on_click=prefill_template_with_demo, type="secondary", disabled=not is_reask_mode(generation_mode),
                  use_container_width=False)
    st.divider()


def construct_reask_text_input(unformatted_prompt):
    if len(st.session_state.trailing_instruction) > 0:
        prompt_suffix = f" {st.session_state.trailing_instruction} Yes or No?"
    else:
        prompt_suffix = f" Yes or No?"
    if len(st.session_state.prompt_text) > 0:
        prompt_prefix = f"{st.session_state.prompt_text} "
    else:
        prompt_prefix = ""
    # Currently always a single space as a separator (in addition to any separators manually added).
    return f"{prompt_prefix}{unformatted_prompt}{prompt_suffix}"


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "[INST] How can I help you?\n"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "[INST] How can I help you?\n"}]
    st.session_state.total_tokens = 10


st.sidebar.button('Clear Chat History', on_click=clear_chat_history, use_container_width=True)


@st.cache_resource(show_spinner=False)
def init_model(model_path):
    return mixtral_batch_classification.load_model(model_path)


mx.random.seed(0)
# Currently, the path is hard-coded. This script is intended as a simple demo, with input formatted for
# Mixtral-8x7B-Instruct-v0.1, rather than a general purpose utility for all MLX LLMs.
with st.spinner('Loading model...Please wait...'):
    model, tokenizer = init_model("mlx_model_quantized")


def generate_response():
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += " " + dict_message["content"]
        else:
            if dict_message["content"].lstrip() == dict_message["content"]:
                string_dialogue += " " + dict_message["content"]
            else:
                string_dialogue += dict_message["content"]
    output = string_dialogue
    prompt = mx.array(tokenizer.encode(output))
    st.session_state.total_tokens += prompt.shape[0]

    tokens = []
    output_string = ""
    attributes = []
    if st.session_state.total_tokens < max_allowed_tokens:
        if generation_mode == kReaskMarker:
            attributes = mixtral_batch_classification.get_document_attributes(model, full_input_tokenized=prompt)
            if attributes[-2] > attributes[-1]:
                output_string = "Yes"
            else:
                output_string = "No"

        else:
            for token, _ in zip(mixtral_batch_classification.standard_generate(prompt, model, 0.0), range(min(max_length, max_allowed_tokens-st.session_state.total_tokens))):
                tokens.append(token)
                st.session_state.total_tokens += 1
                if (len(tokens) % 10) == 0:
                    mx.eval(tokens)
                    eos_index = next(
                        (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_id), None
                    )
                    if eos_index is not None:
                        tokens = tokens[:eos_index]
                    s = tokenizer.decode([t.item() for t in tokens])
                    output_string += s
                    st.session_state.total_tokens += len(tokens)
                    tokens = []
                    if eos_index is not None:
                        break

            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            st.session_state.total_tokens += len(tokens)
            output_string += s

    return output_string, attributes

# User input
if prompt := st.chat_input(disabled=False):
    if generation_mode == kReaskMarker:
        prompt = construct_reask_text_input(prompt)
    prompt += " [/INST]"
    if len(st.session_state.messages) > 2:
        prompt = "[INST] " + prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            response, attributes = generate_response()
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            if generation_mode == kReaskMarker:
                st.subheader(':blue[Reexpression Attributes]')
                attributes = ",".join([str(x) for x in attributes])
                code = f'{attributes}'
                st.code(code, language='json')
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    with st.sidebar:
        st.write(f'Total tokens: {st.session_state.total_tokens}')

    if st.session_state.total_tokens >= max_allowed_tokens:
        st.info(f'The max allowed number of tokens ({max_allowed_tokens}) has been reached. Please click '
                f'"Clear Chat History" to continue.', icon="ℹ️")