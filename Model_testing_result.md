real_data (trained and tested using same data set)
SVM Accuracy:  0.8733624454148472
Perceptron Accuracy:  0.759825327510917
KNN Accuracy:  0.8209606986899564
Random forest Accuracy:  0.9679767103347889  
Neural net accuracy:  0.7976710334788938 

real_data_oct26 without magnetometer data (trained and tested using same data set)
SVM Accuracy:  0.7289272030651341
Perceptron Accuracy:  0.5948275862068966
KNN Accuracy:  0.6369731800766284
Random forest Accuracy:  0.9703065134099617 
Neural net accuracy:  0.8065134099616859

real_data_oct26 with magnetometer (trained and tested using same data set)
SVM Accuracy:  0.7710727969348659
Perceptron Accuracy:  0.64272030651341
KNN Accuracy:  0.6982758620689655
Random forest Accuracy:  0.9645593869731801 
Neural net accuracy:  0.8745210727969349

#### Testing on real data ####
rf_model trained on real_data_oct26, tested on real_data_oct30 
Testing Accuracy:  0.480883322346737

rf_model trained on real_data_oct26, tested on real_data
Testing Accuracy:  0.42410910690717113

rf_model trained on real_data, tested on real_data_oct30
Testing Accuracy:  0.04515491100856955

rf_model trained on real_data, tested on real_data_oct26
Testing Accuracy:  0.21986713065485605


#### Hyperparameter tuning for NN ####
###### Oct 26 data ######
/run1
NN w/ oct26 data w tr label 
MLPClassifier(activation='tanh', alpha=0.016169223265056508,
              hidden_layer_sizes=(32,), max_iter=1000)
Acccuracy: 0.94 

NN w/ oct26 data w/o tr label
MLPClassifier(activation='tanh', alpha=2.689461736717041e-05,
              hidden_layer_sizes=(38,), max_iter=1000)
Acccuracy: 0.96


/run 2
NN w/ oct26 data w tr label 
MLPClassifier(activation='logistic', alpha=0.0001435385074058134,
              hidden_layer_sizes=(56, 113), max_iter=1000)
Acccuracy: 0.95115

NN w/ oct26 data w/o tr label 
MLPClassifier(activation='tanh', alpha=0.006617061009470217,
              hidden_layer_sizes=(36,), learning_rate='adaptive',
              max_iter=1000)
Acccuracy: 0.93675


NN w/ oct26 data w/ only letters 
MLPClassifier(activation='tanh', alpha=0.0003587068208851199,
              hidden_layer_sizes=(108, 108), learning_rate='adaptive',
              max_iter=1000)
Acccuracy: 0.95168 


###### Nov 5 data ######

NN w/ nov5 data w/ tr label
MLPClassifier(alpha=0.00037009903911428525, hidden_layer_sizes=(134, 25),
              learning_rate='adaptive', max_iter=500, solver='lbfgs')
Accuracy: 0.91026

NN w/ nov5 data w/o tr label
MLPClassifier(activation='logistic', alpha=9.571423777674691e-05,
              hidden_layer_sizes=(122, 68), learning_rate='adaptive',
              max_iter=500)
Accuracy: 0.94925

NN w/ nov5 data w/ only letters
MLPClassifier(activation='logistic', alpha=0.025471852345886628,
              hidden_layer_sizes=(23,), learning_rate='adaptive', max_iter=500,
              solver='lbfgs')
Accuracy: 0.90476 