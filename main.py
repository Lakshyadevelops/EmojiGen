import streamlit as st
import replicate
from PIL import Image
import requests
import time
from io import BytesIO
from secret import REPLICATE_API_TOKEN
import os


os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN



# if 'output' not in st.session_state:
    # st.session_state['output'] = None


container = st.container()

container.container()

prompt=st.text_input("Enter emoji search string")
prefix = "a TOK emoji of "
prompt = prefix + prompt
input = {
    "prompt":prompt,
    "apply_watermark":False
}
def call_API():
    output = replicate.run(
        "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e",
        input={
            "prompt":prompt,
            "height":600,
            "width":600,
            "negative_prompt":"worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
        }
    

    )
    output=output[0]
    get_prediction(output)

def get_prediction(output):

    print(output)
    response=requests.get(output)
    print(response)
    container.image(Image.open(BytesIO(response.content)))

col1, col2, col3 = st.columns([1, 6, 0.3])

with col1:
    
    st.button("submit",key='submit')

with col2:
    st.button("Generate Emoji",key='generateButton',on_click=call_API)




