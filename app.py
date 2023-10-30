import streamlit as st
import random
import pickle
import json
import numpy as np
import nltk
import textwrap
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# é uma configuração que reduz o nível de registro do TensorFlow para que ele exiba apenas mensagens importantes e evite inundar a saída com informações detalhadas.
import tensorflow as tf


st.set_page_config(page_title='Chat_Gabi', page_icon='favicon.ico')

# separar as frases em palavras
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents_app.json', encoding='utf-8').read())# abrir/ler o arquivo json

# carregar arquivos
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_app.h5')

# Função para limpar a frase
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Função para converter a frase em um 'saco' de palavras = lista de 0 e 1 que indicam se a palavra está lá ou não
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Função de previsão
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Função para obter a resposta do chatbot
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def obter_resposta(mensagem):
    ints = predict_class(mensagem)
    resposta = get_response(ints, intents)
    return resposta


with st.container():
    st.title('Bem-Vindx ao Chat da Gabi !')
    st.write('---')

c1, c2 = st.columns([2, 2])
d = {}

with st.form('forms'):
    #msg = st.text_input('Sua Mensagem aqui')
    #botao = st.button('Enviar', type='primary', on_click=obter_resposta(msg), args=msg)
    d['mensagem'] = st.text_input('Sua Mensagem aqui:')
    st.form_submit_button('Enviar')
    with c1:
        robo = st.image('imagem4.png')

        if d:
            obter_resposta(d['mensagem'])

        resp = c2.write(f'<style>input[type="text"] {{ height: 150px; }}</style>{obter_resposta(d["mensagem"])}', unsafe_allow_html=True)
