import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow import keras
keras.backend.clear_session()
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st
from keras.models import load_model

st.title("Stock Performance Analysis")


#st.write("Outside the form")

#stock_list = ["RELIANCE.NS", "HINDUNILVR.NS", "TCS.NS", "MSFT", "ADANIPOWER.NS",  "SBIN.NS", "AXISBANK.NS", "AMZN", "TSLA", "CIPLA.NS"]
#selectbox_01 = st.selectbox('Select', stock_list)
#st.form_submit_button(label="Submit", help=None, on_click=None, args=None, kwargs=None)
#user_input = st.text_input('Stock (Supported : RELIANCE.NS, HINDUNILVR.NS, INFY, HDB, TCS, SBIN.NS, TSLA, AMZN, TWTR)', '')


with st.form("my_form"):
    #st.write("Inside the form")
    #slider_val = st.slider("Form slider")
    #checkbox_val = st.checkbox("Form checkbox")
    stock_list = ["RELIANCE.NS", "HINDUNILVR.NS", "TCS.NS", "MSFT", "ADANIPOWER.NS",  "SBIN.NS", "AXISBANK.NS", "AMZN", "TSLA", "CIPLA.NS"]
    selectbox_01 = st.selectbox('Select', stock_list)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    
    if submitted:
    #    st.write("slider", slider_val, "checkbox", checkbox_val)
        try:
            aapl= yf.Ticker(selectbox_01)
            st.write(selectbox_01)
            #data = aapl.history(start="2017-06-15", end="2022-07-15", interval="1d")
            data = yf.download(tickers=selectbox_01,period='5y',interval='1d')
            type(data)
            #st.subheader('Stock Summary')
            #st.write(data.describe())
            #data.head()
            #data.tail()
            opn = data[['Open']]
            opn.plot()
            ds = opn.values
            #ds
            plt.plot(ds)
            normalizer = MinMaxScaler(feature_range=(0,1))
            ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
            #len(ds_scaled), len(ds)
            #len(ds)
            train_size = int(len(ds_scaled)*0.70)
            test_size = len(ds_scaled) - train_size
            #train_size,test_size
            ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]
            #len(ds_train),len(ds_test)
            def create_ds(dataset,step):
                Xtrain, Ytrain = [], []
                for i in range(len(dataset)-step-1):
                    a = dataset[i:(i+step), 0]
                    Xtrain.append(a)
                    Ytrain.append(dataset[i + step, 0])
                return np.array(Xtrain), np.array(Ytrain)
            time_stamp = 100
            X_train, y_train = create_ds(ds_train,time_stamp)
            X_test, y_test = create_ds(ds_test,time_stamp)
            #X_train.shape,y_train.shape
            #X_test.shape, y_test.shape
            X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
            
            model = Sequential()
            model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            model.add(LSTM(units=50,return_sequences=True))
            model.add(LSTM(units=50))
            model.add(Dense(units=1,activation='linear'))
            model.summary()
            model.compile(loss='mean_squared_error',optimizer='adam')
            model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
            
            loss = model.history.history['loss']

            plt.plot(loss)
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            train_predict = normalizer.inverse_transform(train_predict)
            test_predict = normalizer.inverse_transform(test_predict)
            plt.plot(normalizer.inverse_transform(ds_scaled))
            plt.plot(train_predict)
            plt.plot(test_predict)
            type(train_predict)
            test = np.vstack((train_predict,test_predict))

            plt.plot(normalizer.inverse_transform(ds_scaled))
            plt.plot(test)
            #len(ds_test)
            tcount = len(ds_test)
            fcount = tcount - 100
            #fcount
            fut_inp = ds_test[fcount:]
            fut_inp = fut_inp.reshape(1,-1)
            tmp_inp = list(fut_inp)
            #fut_inp.shape
            tmp_inp = tmp_inp[0].tolist()
            lst_output=[]
            n_steps=100
            i=0
            while(i<30):
                if(len(tmp_inp)>100):
                    fut_inp = np.array(tmp_inp[1:])
                    fut_inp=fut_inp.reshape(1,-1)
                    fut_inp = fut_inp.reshape((1, n_steps, 1))
                    yhat = model.predict(fut_inp, verbose=0)
                    tmp_inp.extend(yhat[0].tolist())
                    tmp_inp = tmp_inp[1:]
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    fut_inp = fut_inp.reshape((1, n_steps,1))
                    yhat = model.predict(fut_inp, verbose=0)
                    tmp_inp.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i=i+1
            print(lst_output)
            #len(ds_scaled)
            t_ds_scaled = len(ds_scaled)
            f_ds_scaled = t_ds_scaled - 100
            #f_ds_scaled
            plot_new=np.arange(1,101)
            plot_pred=np.arange(101,131)
            plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[f_ds_scaled:]))
            plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
            ds_new = ds_scaled.tolist()
            #len(ds_new)
            t_ds_new = len(ds_new)
            f_ds_new = t_ds_new - 100
            #f_ds_new
            ds_new.extend(lst_output)
            plt.plot(ds_new[f_ds_new:])
            final_graph = normalizer.inverse_transform(ds_new).tolist()
            st.subheader('Possible performance of Stock')
            fig1 = plt.figure(figsize=(12,6))
            plt.plot(final_graph,)
            plt.ylabel("Price")
            plt.xlabel("Time")
            plt.title("{0} prediction of next month open".format(selectbox_01))
            plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
            plt.legend()
            st.pyplot(fig1)
        except:
            print("Data Load Error")
