import getopt
import os
import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

p='./'
f="iris.csv"
oFile='output.out'

if __name__ == '__main__':
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'm:M:x:X:h', ['m=', 'M=', 'x=', 'X='])
    except getopt.GetoptError as err:
        print('ERROR:', err)
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

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    # Abrir el fichero .cv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)
    # Introducir todos los nombre de las comunas
    ml_dataset = ml_dataset[['Especie', 'Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']]
    categorical_features = []
    #Introducir los nombres de las columnas numéricas
    numerical_features = ['Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']
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

    #Todos los valores que puede tomar la columna 'ESPECIE'
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map)
    del ml_dataset['Especie']

    #Eliminar filas para las que se desconoce el destino
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    ##############################train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
    train, test = train_test_split(ml_dataset, test_size=0.2, random_state=42, stratify=ml_dataset[['__target__']])
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    #Valores a imputar
    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'},
                       {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'},
                       {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
                       {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    #Elimina las filas con missing records
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    #Rellena las filas con missing records con las medias
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

    #Valores a reescalar (entre las columnas)
    rescale_features = {'Ancho de sepalo': 'AVGSTD', 'Largo de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD',
                        'Ancho de petalo': 'AVGSTD'}
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


    X_train = train.drop('__target__', axis=1)
    X_test = test.drop('__target__', axis=1)

    y_train = np.array(train['__target__'])
    y_test = np.array(test['__target__'])

    indice = 1
    F1 = []
    F1_max = 0
    Total = []
    for t1 in range(int(m), int(M) + 1):
        for t2 in range(int(x), int(X) + 1, 3):
            clf = tree.DecisionTreeClassifier(
                random_state=42,
                criterion='gini',
                splitter='best',
                max_depth=t2,
                min_samples_leaf=t1)
            #Indica la estrategia que va a utilizar
            clf.class_weight = "balanced"
            #Entrena con el modelo indicado
            clf.fit(X_train,y_train)
            #Ya está entrenado
            #Recoge las predicciones
            predictions = clf.predict(X_test)
            probas = clf.predict_proba(X_test)
            predictions = pd.Series(data=predictions, index=X_test.index, name='predicted_value')
            cols = [
                u'probability_of_value_%s' % label
                for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
            ]
            probabilities = pd.DataFrame(data=probas, index=X_test.index, columns=cols)
            # Build scored dataset
            results_test = X_test.join(predictions, how='left')
            results_test = results_test.join(probabilities, how='left')
            results_test = results_test.join(test['__target__'], how='left')
            results_test = results_test.rename(columns={'__target__': 'Especie'})
            i = 0
            for real, pred in zip(y_test, predictions):
                i += 1
                if i > 5:
                    break

            print(f1_score(y_test, predictions, average='macro'))
            print(classification_report(y_test, predictions))
            print(confusion_matrix(y_test, predictions, labels=[1, 0]))
            #tree.plot_tree(clf)
            info = [
                'caso: ' + str(indice) + ': ' + 'min= ' + str(t1) + ' max= ' + str(t2) + '-->',
                f1_score(y_test, predictions, average='macro'),
                classification_report(y_test, predictions),
                confusion_matrix(y_test, predictions, labels=[1, 0])
            ]
            F1.append(info[1])
            Total.append([info[0],info[1],info[2]])
            indice = indice + 1
            for frase in info:
                if (os.listdir("./TREE") == []):
                    file = open("./TREE/iris_TREE.csv", "w")
                else:
                    file = open("./TREE/iris_TREE.csv", "a")
                file.write("\n" + str(frase) + "\n")
    F1_max = max(F1)
    for caso in Total:
        if(caso[1] == F1_max):
            file2 = open("./TREE/mejor_modelo_Iris.csv", "a")
            file2.write("\n" + str(caso[0]) + " macro f1-score: " + str(caso[1]) + "\n" + str(caso[2]) + "\n")
print("FINAL")