#!-*-coding:utf8-*-
import sys
from sklearn.model_selection  import train_test_split
from sklearn.datasets         import load_iris
from sklearn.neural_network   import MLPClassifier
from sklearn.grid_search      import GridSearchCV
from sklearn.metrics          import classification_report
from numpy                    import array

iris = load_iris()
iris_d = iris['data']
targets = []
for v in iris['target']:
    z = [0,0,0]
    z[v] = 1
    targets.append(z)
iris_t = array(targets)
Xtr, Xts, Ytr, Yts = train_test_split( iris_d, iris_t )

if(sys.argv[1]=='busca'):
## Ajuste de parametros
    params = {"alpha" : [0.1, 0.01, 0.001], "max_iter" : [50, 100, 200],\
              "batch_size" : [5, 10, 20], "activation" : ['relu','tanh'],\
              "hidden_layer_sizes": [10, 20, 50, 100]}
    clf = MLPClassifier()
    gs = GridSearchCV( clf, params, n_jobs=2, verbose=1, scoring='precision_macro' )
    gs.fit(Xtr,Ytr)
    print(gs.best_params_)
elif(sys.argv[1]=='entrena'):
# El resultado de esto son los siguientes parámetros
    params = {'alpha': 0.01, 'activation': 'relu', 'max_iter': 200, 'batch_size': 10,\
              'hidden_layer_sizes': 50}
    clf = MLPClassifier()
    clf.set_params(**params)
    clf.fit(Xtr, Ytr)
    for v, p in zip(Xts, Yts):
        print "{0} vs {1}".format( clf.predict_proba(v.reshape(1,-1)), p )
    print("Reporte")
    print(classification_report(Yts, clf.predict(Xts)))
