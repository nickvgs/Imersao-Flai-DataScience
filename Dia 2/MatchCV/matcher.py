import streamlit as st
import pdfplumber
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(page_title = 'MatchCV - Powered by FLAI',  
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')


def NuvemDePalavras(textoCRU, salvar = True):
    '''
    textoCRU: texto extraido de um PDF
    ''' 
    lista_de_palavras = nltk.tokenize.word_tokenize(textoCRU)  # transforma o texto cru em uma lista de termos
    lista_de_palavras = [palavra.lower() for palavra in lista_de_palavras]  # deixando tudo minusculo

    keywords = [palavra for palavra in lista_de_palavras if not palavra in stop_words and not palavra in pontuacao]  # tira as pontuacoes e stopwords 
    textocv = " ".join(s for s in keywords)  # junta tudo em um texto só novamente. 


    wordcloud = WordCloud(background_color = '#0f54c9', 
                        max_font_size = 150, 
                        width = 1280, 
                        height = 720, 
                        colormap= 'Blues').generate(textocv) 

    # mostrar a imagem final
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud)
    ax.set_axis_off()
    plt.imshow(wordcloud)

    if salvar:
        wordcloud.to_file("wordcloud.png")

    plt.show()


def MatchCV(textoCRU, vaga, limite = 3):
    '''
    textoCRU: texto extraido de um PDF
    vaga: dicionario de palavras-chave e pesos
    '''
    lista_de_palavras = nltk.tokenize.word_tokenize(textoCRU)  # transforma o texto cru em uma lista de termos
    lista_de_palavras = [palavra.lower() for palavra in lista_de_palavras]  # deixando tudo minusculo

    keywords = [palavra for palavra in lista_de_palavras if not palavra in stop_words and not palavra in pontuacao]  # tira as pontuacoes e stopwords 
    textocv = " ".join(s for s in keywords)  # junta tudo em um texto só novamente. 

    pesos = list(vaga.values())
    palavras_chaves = list(vaga.keys())

    cont = [textocv.count(pc) for pc in palavras_chaves]  # conta quantas vezes cada termo da vaga aparece no texto do cv

    def aux(x, limite):
        return x if x <= limite else limite

    cont = [aux(i, limite) for i in cont]   # coloca o limite na contagem de palavras

    pmax = np.sum(np.array(pesos) * limite) 

    score = ((np.array(cont) * pesos).sum()/pmax).round(4)

    return score


nltk.download('punkt')
nltk.download('stopwords')
pontuacao = ['(', ')', ';', ':', '[', ']', ',']
stop_words = nltk.corpus.stopwords.words('portuguese')
 

ds_senior = {'python': 1,
             'linguagem r': 1,
             'sql': 1,
             'machine learning': 2,
             'estatística': 2, 
             'big data': 2,
             'negócio': 2}


st.markdown('# **MatchCV FLAI**')
st.markdown('## Otimizando a eficiência de RHs desde hoje')
st.markdown('---')
st.markdown('## Carregue o PDF de um CV que você deseja analizar')

PDFs = st.file_uploader('O arquivo deve ter no máximo 10mb e ter a extenção ".pdf"',
				 type = ['pdf'],
				 accept_multiple_files = True)

if st.button('Carregar'):

	for arquivoPDF in PDFs:
		if arquivoPDF is not None:
			cv = pdfplumber.load(arquivoPDF)
			primeira_pagina = cv.pages[0]
			textoCRU = primeira_pagina.extract_text()
		
		st.markdown('---')
		st.write(textoCRU[:200])

		score = MatchCV(textoCRU, ds_senior)

		frase = 'O Match desse CV com essa vaga foi de {}'.format(score)
		st.markdown('## **' + frase + '**')
		NuvemDePalavras(textoCRU)
		st.image('./wordcloud.png')

 