# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#-----------------------------------------------------------------------------------------------------KNN--------------------------------------------------------------------------------------------------------
p = './'
f = 'nombreFichero.csv'

if __name__ == '__main__':
	try:
		options, remainder = getopt.getopt(sys.argv[1:],'k:K:d:D:',['k=','K=','d=','D='])
	exception getopt.GetoptError as err:
		print('Error: ', err)
		sys.exit(1)
	for opt, arg in options:
		if opt == '-k':
			k = arg
		elif opt == '-K':
			K = arg
		elif opt == '-d':
			d = arg
		elif opt == '-D':
			D = arg
	iFile = p + str(f)
	#Función coerce_to_unicode(x)
	#Abrir el fichero y cargarlo 
	ml_dataset = pd.read_csv(iFile)
	#Introducir nombres de las columnas --> 
    ml_dataset = ml_dataset[]
	#Introducir nombres de las columnas numéricas --> 
    categorical + numeriacl + text + fors
	#Introducir todos los valores que puede tomar la columna 'COL' --> 
    target_map + ml_dataset + del
	#Eliminar filas para las que se desconoce el destino --> 
    ml_dataset.isnull 
	#Definir train y test --> 
    train,test = train_test_split()
	#Valores a imputar --> 
    drop + impute
	#Eliminar filas con missing records --> 
    for feature in drop
	#Rellenar huecos de los missing records con la media --> 
    for feature in impute
	#Reescalar valores (entre las distintas columnas) --> 
    reescale_features + for
	#Definir train y test de X e Y 
	
	#Si es numérico balancear (valores de su columna)
	undersample = RandomUnderSampler(sampling_strategy = 0.5)
	trainXUnder, trainYUnder = undersample.fit_resample(X_train, y_train)
	testXUnder, trainYUnder = undersample.fit_resample(X_test, y_test)
	
	w = ['distance','uniform']
	F1 = []
	Total = []
	F1_max = 0
	indice = 1
	for t1 in w:
		for t2 in range(int(d), int(D) + 1):
			for t3 in range(int(k), int(K) + 1):
				if t3 % 2 != 0:
					#Definir el algoritmo que va a utilizar --> clf = KNeighborsClassifier()
					#Indicar la estrategia a utilizar --> clf.class_weight
					#Entrenar 
					clf.fit(trainXUnder, trainYUnder)
					
					#Ya está entrenado 
					#Recoger predicciones 
					predictions = clf.predict(testXUnder)
					probas = clf.predict_proba(testXUnder)
					
					#Crearlas --> predictions + cols + probabilities
					#Recoger resultados --> results + for
					#Crear array y datos para el fichero 
					info[
					   'caso: ' + str(indice) + ': d= ' + str(t2) + ' k=' + str(t3) + ' weight= ' + t1,
					   f1_score(testYUnder, predictions, averages = 'macro'), ##SI NO ES BINARIO NONE SIN COMILLAS
					   classification_report(testYUnder, predictions),
					   confusion_matrix(testYUnder, predictions, labels=[1,0])]
					F1.append = info[1]
					Total.append = [info[0], info[1], info[2]]
					for frase in info:
						file = open('nombrecarpeta/nombrearchivo.csv', 'a')
						file.write('\n' + str(frase) + '\n')
	F1_max = max(F1) ##SI NO ES BINARIO 
	F1_max = max(F1[0]) ##SI ES BINARIO 
	for caso in Total:
		f1 = caso[1] ##SI ES BINARIO
		if f1 [0] == F1_max: ##SI ES BINARIO
		if caso[1] == F1_max: ##SI NO ES BINARIO 
			file2 = open('nombrecarpeta/nombrearchivo.csv','a')
			file2.write('\n' + str(caso[0]) + ', macro average = ' + str(caso[1]) + '\n' + str(caso[2]))

print('FINAL')




#----------------------------------------------------------------------------------------------------TREE--------------------------------------------------------------------------------------------------------
p = './'
f = 'nombrefichero.csv'

if __name__ == '__main__':
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'm:M:x:X:', ['m=', 'M=', 'x=', 'X='])
    except getopt.GetoptError as err:
        print('ERROR: ' + err)
        sys.exit(1)
    
    for opt, arg in options:
        if opt == '-m':
            m = arg
        elif opt == '-M':
            M = arg
        elif opt == '-x':
            x = arg
        elif opt == '-X':
            X = arg
    iFile = p + str(f)
    
    #Función coerce_to_unicode(x)
	#Abrir el fichero y cargarlo 
	ml_dataset = pd.read_csv(iFile)
	#Introducir nombres de las columnas --> ml_dataset = ml_dataset[]
	#Introducir nombres de las columnas numéricas --> categorical + numeriacl + text + fors
	#Introducir todos los valores que puede tomar la columna 'COL' --> target_map + ml_dataset + del
	#Eliminar filas para las que se desconoce el destino --> ml_dataset.isnull 
	#Definir train y test --> train,test = train_test_split()
	#Valores a imputar --> drop + impute
	#Eliminar filas con missing records --> for feature in drop
	#Rellenar huecos de los missing records con la media --> for feature in impute
	#Reescalar valores (entre las distintas columnas) --> reescale_features + for
	#Definir train y test de X e Y 
	
	#Si es numérico balancear (valores de su columna)
	undersample = RandomUndersampler(sampling_strategy = 0.5)
	trainXUndersample, trainYUndersample = undersample.fit_resample(X_train, y_train)
	testXUndersample, testYUndersample = undersample.fit_resample(X_test, y_test)
	
	
    
    
    