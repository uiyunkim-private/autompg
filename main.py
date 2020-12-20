





#
# model = SGDRegressor()
# model.fit(X_train,y_train)
# print(model.score(X_train,y_train))
# print(model.score(X_test,y_test))
# y_pred = model.predict(X_test)
#
# plt.plot(y_test)
# plt.plot(y_pred)
# plt.show()
#
#
# print(y_test)
# print(y_pred)
#
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop

def keras_trainer(X_train,y_train):
    def dense_model():
        model = Sequential()
        model.add(Dense(128,activation='relu',input_dim=9))
        model.add(Dense(128,activation='relu'))

        # model.add(Dense(128,kernel_initializer='he_uniform'))
        model.add(Dense(1,activation='linear'))
        model.compile(optimizer=RMSprop(),loss='mse',metrics=['mse'])
        return model

    model = dense_model()

    history = model.fit(x=X_train,y=y_train,batch_size=4,epochs=100,validation_split=0.2)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()

    return model
model = keras_trainer(data,label)
y_pred = model.predict(X_test)

def plot_regressor(y_pred,y_test):
    plt.plot(range(0,2000), range(0,2000))
    lim = max([max(y_pred),max(y_test)])
    plt.scatter(y_pred,y_test)
    plt.xlim(0,lim)
    plt.ylim(0,lim)
    plt.show()

plot_regressor(y_pred,y_test)