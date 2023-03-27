# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import os
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

k=1
d=1
p='./'
f="iris.csv"
oFile="output.out"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'k:K:d:D:s:',['k=','K=','d=','D=','s='])
        

    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt == '-k':
            k = arg
        elif opt == '-K':
            K = arg
        elif opt == '-d':
            d = arg
        elif opt == '-D':
            D = arg
        elif opt == '-s':
            s = arg
        
    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected


    #  Se pasan todos los atributos de texto a unicode
    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))
    
    # Se introducen los nombres de las columnas
    ml_dataset = ml_dataset[
        ['Largo de sepalo','Ancho de sepalo','Largo de petalo','Ancho de petalo','Especie']]

    # Se guardan los nombres de las columnas de valores categóricos
    categorical_features = ['Especie']
    # Se guardan los nombres de las columnas de valores numericos
    numerical_features = ['Largo de sepalo','Ancho de sepalo','Largo de petalo','Ancho de petalo']
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')


    # Los valores posibles de la clase a predecir Especie. 
    # Puede ser de 3 clases
    target_map = {'Iris-versicolor': 0, 'Iris-setosa': 1, 'Iris-virginica': 2}

    # Columna que se utilizará
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map)
    del ml_dataset['Especie']

    # Se borran las filas en las que la clase a predecir no aparezca
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    # Se crean las particiones de Train/Test
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    # Lista con los atributos que cuando faltan en una instancia hagan que se tenga que borrar
    drop_rows_when_missing = []

    # Lista con los atributos que cuando faltan en una instancia se tenga que corregir haciendo la media, mediana, etc. del resto
    impute_when_missing = [{'feature': 'Largo de sepalo', 'impute_with': 'MEAN'},
                           {'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'},
                           {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
                           {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]
                           
    # Se borran las filas en las que falten los atributos que haya en la lista drop_rows_when_missing
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Se corrigen todos los datos faltantes de los atributos en la lista impute_when_missing dependiendo de como se deban tratar
    # En este caso todos se corrigen con la media del resto de instancias
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


    # Lista con los métodos para escalar cada atributo numerico 
    rescale_features = {'Largo de sepalo': 'AVGSTD', 
                        'Ancho de sepalo': 'AVGSTD', 
                        'Largo de petalo': 'AVGSTD',
                        'Ancho de petalo': 'AVGSTD'}
    
    # Se reescala
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    


    # Valores del conjunto Train
    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']
    
    # Valores del conjunto Test
    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    # Etiquetas del conjunto Train
    trainY = np.array(train['__target__'])
    # Etiquetas del conjunto Test
    testY = np.array(test['__target__'])

    # Explica lo que se hace en este paso
    # Se realiza undersampling con la funcion de la libreria imbalanced-learn.
    # El undersampling consiste en borrar instancias de la clase dominante para equilibrar el dataset

    # Utilizamos un dict como sampling strategy
    sampling_strategy = {0: 10, 1: 10, 2: 10}
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)

    # Se reemplazan los conjuntos Train/Test con unos conjuntos a los que se les ha realizado undersampling
    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # Se probara con las dos formas de calcular los pesos
    weights = ["uniform", "distance"]
    # Guardo las iteraciones
    # Formato: [n_iteracion, fScore, classification_report, k_neighbors, valorD, w, ]
    iteraciones = []
    n_iteracion = 0

    fScoresMicro = []
    fScoresMacro = []
    fScoresWeighted = []

    # Escribo las columnas en el fichero
    with open('./knn/iris_KNN.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['n_iteracion', 'fScoreMacro', 'fScoreMicro', 'fScoreWeighted', 'precision', 'recall', 'k_neighbors', 'valorD', 'w'])
    

    for w in weights:
        for valorD in range(int(d), int(D) +1):
            for valorK in range(int(k), int(K) +1):
                # Solo hacer si los vecinos son impares
                if valorK % 2 != 0:
                    
                    # Se crea el modelo KNN
                    clf = KNeighborsClassifier(n_neighbors = valorK, weights = w, algorithm = "auto", leaf_size = 30, p = valorD)

                    # Se modifica el metodo de utilizar los pesos de KNN
                    clf.class_weight = "balanced"

                    # Finalmente, se entrena el modelo con la particion de trainUndersampled
                    clf.fit(trainXUnder, trainYUnder)

                    # Ya se ha entrenado el modelo
                    predictions = clf.predict(testXUnder)
                    probas = clf.predict_proba(testXUnder)

                    predictions = pd.Series(data=predictions, index=testXUnder.index, name='predicted_value')
                    cols = [
                        u'probability_of_value_%s' % label
                        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                    ]
                    probabilities = pd.DataFrame(data=probas, index=testXUnder.index, columns=cols)

                    
                    results_test = testXUnder.join(predictions, how='left')
                    results_test = results_test.join(probabilities, how='left')
                    results_test = results_test.join(test['__target__'], how='left')
                    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

                    i=0
                    for real,pred in zip(testYUnder,predictions):
                        #print(real,pred)
                        i+=1
                        if i>5:
                            break

                    # Se obtiene la precision y recall
                    # output_dict = True para que devuelva un dict
                    classific_report = classification_report(testYUnder,predictions, output_dict=True)
                    precision = classific_report['0']['precision']
                    recall = classific_report['0']['recall']
                    # Formato: [n_iteracion, fScore, classification_report, k_neighbors, valorD, w,]
                    iteracion = {'n_iteracion': n_iteracion,
                                # average=None porque es un problema de clasificacion binaria
                                 'fScoreMacro': f1_score(testYUnder, predictions, average='macro'),
                                 'fScoreMicro': f1_score(testYUnder, predictions, average='micro'),    
                                 'fScoreWeighted': f1_score(testYUnder, predictions, average='weighted'),  
                                 'precision': precision,
                                 'recall': recall,       
                                 'k_neighbors': valorK,
                                 'valorD': valorD,
                                 'w': w}
                    
                    # Guardo los resultados de esta iteracion
                    iteraciones.append(iteracion)

                    # Guardo cada fScore de esta iteración para al final coger el mejor modelo según el tipo de fScore
                    fScoresMacro.append(iteracion['fScoreMacro'])
                    fScoresMicro.append(iteracion['fScoreMicro'])
                    fScoresWeighted.append(iteracion['fScoreWeighted'])

                    # Aumento el n_iteracion
                    n_iteracion += 1

                    # matriz_confusion = confusion_matrix(testYUnder, predictions, labels=[1,0])

                    # Escribo en el fichero los resultados
                    with open('./knn/iris_KNN.csv', 'a') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        filewriter.writerow([iteracion['n_iteracion'], iteracion['fScoreMacro'], 
                                            iteracion['fScoreMicro'], iteracion['fScoreWeighted'], 
                                            iteracion['precision'], iteracion['recall'],
                                            iteracion['k_neighbors'], iteracion['valorD'], 
                                            iteracion['w']])

                    
                    #print(f1_score(testYUnder, predictions, average=None))
                    #print(classification_report(testYUnder,predictions))
                    #print(confusion_matrix(testYUnder, predictions, labels=[1,0]))

    # Terminadas las iteraciones obtenemos el mejor fScore
    fScoreMacro_max = max(fScoresMacro)
    fScoreMicro_max = max(fScoresMicro)
    fScoreWeighted_max = max(fScoresWeighted)

    
    for iteracion in iteraciones:
        # Si estoy en el modelo con el mejor fScoreMacro
        if iteracion['fScoreMacro'] == fScoreMacro_max:
            mejorMacro = iteracion
        # Si estoy en el modelo con el mejor fScoreMicro
        if iteracion['fScoreMicro'] == fScoreMicro_max:
            mejorMicro = iteracion
        # Si estoy en el modelo con el mejor fScoreWeighted
        if iteracion['fScoreWeighted'] == fScoreWeighted_max:
            mejorWeighted = iteracion

    # Se escriben los mejores modelos segun el tipo de fScore
    f = open("./knn/best_FScore_iris.csv", "w")
    f.write("\nMacroFscore" + str(mejorMacro) + "\nMicroFscore" + str(mejorMicro) + "\nWeightedFscore" + str(mejorWeighted))




# Se guarda el modelo con el mejor fScore
import pickle
nombreModel = "modeloKNNconPreproceso.sav"

# Dependiendo del parametro -score introducido se guarda un modelo u otro
if(s == "macro"):
    valorK = mejorMacro['k_neighbors']
    w = mejorMacro['w']
    valorD = mejorMacro['valorD']
    print("Se guarda el mejor modelo segun la macro Fscore")
elif (s == 'micro'):
    valorK = mejorMicro['k_neighbors']
    w = mejorMicro['w']
    valorD = mejorMicro['valorD']
    print("Se guarda el mejor modelo segun la micro Fscore")
elif (s == 'weighted'):
    valorK = mejorWeighted['k_neighbors']
    w = mejorWeighted['w']
    valorD = mejorWeighted['valorD']
    print("Se guarda el mejor modelo segun la weighted Fscore")
    
# Se crea el modelo a guardar
# El modelo sera el mejor segun la fscore elegida

clf = KNeighborsClassifier(n_neighbors = valorK, weights = w, algorithm = "auto", leaf_size = 30, p = valorD)
# Se modifica el metodo de utilizar los pesos de KNN
clf.class_weight = "balanced"
# Finalmente, se entrena el modelo con la particion de trainUndersampled
clf.fit(trainXUnder, trainYUnder)


saved_model = pickle.dump(clf, open(nombreModel,'wb'))
print('Modelo guardado correctamente empleando Pickle')
      

print("bukatu da")