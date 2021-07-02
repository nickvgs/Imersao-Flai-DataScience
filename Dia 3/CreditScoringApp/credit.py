import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


st.set_page_config(page_title = 'Credit Scoring - Powered by FLAI',  
				   layout = 'wide', 
				   initial_sidebar_state = 'auto')




st.title('Credit Scoring App')

st.sidebar.title('Credit APP') 

st.markdown('---')
st.markdown('### Entrada das variáveis do **proponente ao crédito**:')
st.markdown(' ')

col1, col2, col3, col4 = st.beta_columns(4)

x1 = col1.radio('Conta', ['negativo', '[0-200)', '200+', 'sem conta'], help = 'Essa variável é tchululu tchalala')
x3 = col1.radio('Historico', ['primeira vez', 'creditos quitados', 'pagamento em dia', 
       'já atrasou pagamentos', 'conta crítica'])
x4 = col1.radio('Motivo', ['carro novo', 'carro usado', 'móveis', 
       'radio/televisão', 'itens de casa', 'reparos', 
       'educação', 'férias', 'retreinamentos', 
       'negócios', 'outros'])
x6 = col3.radio('Poupança', ['<100', '[100-500)', '[500-1000)',
       '>1000', 'sem conta'])
x7 = col2.radio('Emprego', ['desempregado', '< 1 ano',  '[1,4) anos',
        '[4,7) anos', '> 7 anos'])

x8 = col2.radio('Taxa', [1,2,3, 4])
x9 = col4.radio('Status', ['masculino/divorciado', 'feminino/divorciado',
       'masculino/solteiro', 'masculino/casado'])
x10 = col2.radio('Garantia', ['nenhum', 'co-aplicante', 'fiador'])
x11 = col2.radio('Residencia', [1,2,3,4])
x12 = col2.radio('Propriedades', ['imobiliario', 'seguro  de vida',
      'carro', 'sem propriedades'])

x14 = col3.radio('Financiamentos', ['bancos', 'lojas', 'nenhum'])
x15 = col3.radio('Moradia', ['alugada', 'própria', 'de graça'])
x16 = col3.radio('Creditos', [1, 2, 3, 4])
x17 = col3.radio('Trabalho', ['desempregado',  'nível 1',  'nível 2',  'nível 3' ])
x18 = col1.radio('Dependentes', [1,2])

x19 = col4.radio('Telefone', [ 'não', 'sim'])
x20 = col4.radio('Estrangeiro', ['não', 'sim'])
x13 = col4.slider('Idade', 18, 80, 25, 1)
x2 = col4.slider('Duração', 3, 72, 12, 1)
x5 = col4.slider('Quantia', 250, 25000, 1000, 50)



st.markdown('---')
 
dicionario  =  {'conta': [x1],
				'duração': [x2],
				'historico': [x3],
				'motivo': [x4],
				'quantia': [x5],
				'poupança': [x6],
				'emprego': [x7],
				'taxa': [x8],
				'status': [x9],
				'garantia': [x10],
				'residencia': [x11],		
				'propriedades': [x12],
				'idade': [x13],
				'financiamentos': [x14],
				'moradia': [x15],
				'creditos': [x16],
				'trabalho': [x17],
				'dependentes': [x18],
				'telefone': [x19],
				'estrangeiro': [x20]}

dados = pd.DataFrame(dicionario) 

st.write(dados)

st.markdown('---')

modelo = load_model('modelo-german-credit-data')



if st.button('EXECUTAR O MODELO'):
	saida = predict_model(modelo, dados)
	prob = float(saida['Score'])
	clas = int(saida['Label']) 

	if clas == 0:
		prob = 1 - prob
		st.markdown('### Previsão do Modelo: **Bom Pagador**, com score = {}'.format(round(prob, 2)))  
	else:
		st.markdown('### Previsão do Modelo: **Mau Pagador**, com score = {}'.format(round(prob, 2)))     

	if prob < 0.44:  
		st.success('Usuário na Faixa de Score A - APROVADO SEM RESTRIÇÕES')
	elif prob < 0.50:
		st.info('Usuário dnaa Faixa de Score B - APROVADO COM RESTRIÇÕES')
	elif prob < 0.55:
		st.error('Usuário na Faixa de Score C - CONVERSAR COM O GERENTE')
	else:
		st.error('Usuário na Faixa de Score D - NEGAR CRÉDITO/REVER CONDIÇÕES')



	#pred = float(saida['Label'].round(2)) 
	#pred = saida['Score']
	#s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred) 
	#st.markdown('### **' + pred + '**')  