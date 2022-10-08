#!pip install sklearn

#2021/1
#Features: 0 -> não, 1 -> sim
#saía de casa com freq?
#tomou vacina?
#tomou cloroquina?
#usava máscara?

contraiu1 = [1,0,1,0]
contraiu2 = [1,0,0,1]
contraiu3 = [0,1,1,0]
contraiu4 = [1,0,1,1]

naocontraiu1 = [0,1,0,1]
naocontraiu2 = [1,1,0,1]
naocontraiu3 = [0,0,1,1]
naocontraiu4 = [0,1,0,0]

treino_x = [contraiu1, contraiu2, contraiu3, contraiu4, naocontraiu1, naocontraiu2, naocontraiu3, naocontraiu4]
treino_y = [1,1,1,1,0,0,0,0]

#Linear Support Vector Classification: método de aprendizado supervisionado
from sklearn.svm import LinearSVC

#??LinearSVC
model = LinearSVC()
model.fit(treino_x, treino_y)

misterio1 = [0,1,1,1] 
misterio2 = [1,0,0,0]
misterio3 = [1,1,1,1]

teste_x = [misterio1, misterio2, misterio3]
teste_y = [0,1,0]

#testando o nosso modelo
previsoes = model.predict(teste_x)
print(previsoes)

#Calcular acurácia do modelo manualmente:
corretos = (previsoes == teste_y).sum()
total = len(teste_y) 
taxa_de_acerto = corretos/total
print('A taxa de acerto foi de: %.2f%%' % (taxa_de_acerto*100))

#Calcular acurácia usando sklearn
from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print('A taxa de acerto foi de: %.2f%%' % (taxa_de_acerto*100))
