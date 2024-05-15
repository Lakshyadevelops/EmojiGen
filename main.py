from emoji_data import EmojiSequence
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import replicate
from PIL import Image,ImageDraw,ImageFont
import requests
from io import BytesIO
from secret import REPLICATE_API_TOKEN
import os
import numpy as np
import pandas as pd
import gensim
print("imports done")


os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
st.markdown('<p class="headline">Emoji Fi-Gen</p>', unsafe_allow_html=True)

st.markdown("""<style>.big-font {font-size:150px !important;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>.headline {text-align:center;font-size:40px}</style>""", unsafe_allow_html=True)

container = st.container()
container.container()

def call_API():
    output = replicate.run(
        "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e",
        input={
            "prompt":model_prompt,
            "height":800,
            "width":800,
            "negative_prompt":"worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
        }
    
    )
    output=output[0]
    response=requests.get(output)
    container.empty()
    container.image(Image.open(BytesIO(response.content)))

prompt=st.text_input("Enter emoji search string")
prefix = "a TOK emoji of "
model_prompt = prefix + prompt
input = {
    "prompt":model_prompt,
    "apply_watermark":False
}

def process_words(text):
    words = text.split(" ")
    word_vectors = [ model.vectors_norm[model.vocab[word].index] for word in words if word in model.vocab]
    if len(word_vectors) > 0:
        mean_vector = np.array(word_vectors).mean(axis=0)
        unit_vector = gensim.matutils.unitvec(mean_vector).astype(np.float32).tolist()
    else:
        unit_vector = np.zeros(model.vector_size, ).tolist()
    return unit_vector


# LOAD MODEL
st.cache_resource()
def loadModel():
    MODEL_DIRECTORY="binary_stuff/GoogleNews-vectors-negative300-SLIM.bin.gz"
    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_DIRECTORY, binary=True)
    model.init_sims(replace=True)
    print("loaded model")
    return model
model=loadModel()


# CREATE DATAFRAME
@st.cache_data()
def createDataFrame():
    d = {'emoji': [], 'description': []}
    for (emoji, emoji_meta) in EmojiSequence:
        d['emoji'].append(emoji)
        d['description'].append(emoji_meta.description)

    df = pd.DataFrame(d)
    df = df[df['description'] != '']

    df['description'] = df['description'].str.split(' skin tone').str[0].str.replace(':', '').str.replace(',', '')
    df = df.drop_duplicates(subset=['description'])
    df['vector'] = df['description'].apply(process_words)
    print("created data frame")
    return df
df=createDataFrame()


def find_similarity_to_search(search_vector):
    def func(emoji_vector):
        b_emoji = np.array(emoji_vector)
        cos_sim = np.dot(search_vector, b_emoji) / (np.linalg.norm(search_vector) * np.linalg.norm(b_emoji))
        return cos_sim
    return func


THRESHOLD = 0.7
def searchEmoji(df,search_text):
    search_vector = np.array(process_words(search_text))
    similarity_search = find_similarity_to_search(search_vector)
    df['similarity'] = df['vector'].apply(similarity_search)
    df2 = df[ df['similarity'] > THRESHOLD ]
    df2=df2.sort_values(['similarity'],ascending=False)
    x = df2.nlargest(5, 'similarity')
    emojiList = ""
    for emoji in x['emoji']:
        emojiList +=emoji
    container.empty()
    container.markdown('<p class="big-font">'+emojiList+'</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 6, 0.3])
with col1:
    st.button("Find Emoji",on_click=searchEmoji(df,prompt))

with col2:
    st.button("Generate Emoji",on_click=call_API)