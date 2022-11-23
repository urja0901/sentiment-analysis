from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from sklearn.metrics import  confusion_matrix
from tensorflow.keras.optimizers import RMSprop


max_len = 500

def tensorflow_based_model(): #Defined tensorflow_based_model function for training tenforflow based model
    inputs = Input(name='inputs',shape=[max_len])#step1
    layer = Embedding(2000,50,input_length=max_len)(inputs) #step2
    layer = LSTM(64)(layer) #step3
    layer = Dense(256,name='FC1')(layer) #step4
    layer = Activation('relu')(layer) # step5
    layer = Dropout(0.5)(layer) # step6
    layer = Dense(1,name='out_layer')(layer) #step4 again but this time its giving only one output as because we need to classify the tweet as positive or negative
    layer = Activation('sigmoid')(layer) #step5 but this time activation function is sigmoid for only one output.
    model = Model(inputs=inputs,outputs=layer) #here we are getting the final output value in the model for classification
    return model #function returning the value when we call it

def train_model(X_train, Y_train, X_test=None, Y_test=None):
    metadata = {}
    model = tensorflow_based_model() # here we are calling the function of created model
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])  
    history=model.fit(X_train,Y_train,batch_size=128,epochs=10, validation_split=0.1)# here we are starting the training of model by feeding the training data
    acc = model.evaluate(X_test,Y_test) #we are starting to test the model here
    print(model.metrics)
    y_pred = model.predict(X_test) #getting predictions on the trained model
    y_pred = (y_pred > 0.5) 
    CR=confusion_matrix(Y_test, y_pred)
    metadata['history'] = history
    metadata['accuracy'] = acc
    metadata['confusion_matrix'] = CR

    return model, metadata


