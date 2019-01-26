from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import csv
from flask import Flask, jsonify

inputs_ = []
outputs_ = []
inputs_testing = []

#GET INPUTS FOR TRAINING
with open('Inputs-training.csv', 'r') as f:
  in_reader = csv.reader(f, delimiter=';')
  for row in in_reader:
      inputs_.append([float(row[0]), float(row[1])])
      
#GET OUTPUTS FOR TRAINING
with open('Outputs-training.csv', 'r') as f:
  out_reader = csv.reader(f, delimiter=';')
  for row in out_reader:
      outputs_.append([float(row[0]), float(row[1])])
      
#GET INPUTS FOR TESTING
with open('Inputs-testing.csv', 'r') as f:
  out_reader = csv.reader(f, delimiter=';')
  for row in out_reader:
      inputs_testing.append([float(row[0]), float(row[1])])
      
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
scaler_2 = MinMaxScaler(feature_range=(-1.0, 1.0))
scaler_3 = MinMaxScaler(feature_range=(-1.0, 1.0))

inputs_fit = scaler.fit_transform(inputs_)
outputs_fit = scaler_2.fit_transform(outputs_)
inputs_testing = scaler_3.fit_transform(inputs_testing)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


app = Flask(__name__)
 
@app.route("/")
def predict():
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae', 'acc'])
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])
    
    # Fit the model
    print(loaded_model.predict(inputs_fit[699:999], batch_size=None, verbose=0, steps=2)[:5])
    
    print('Estimated:')
    print(scaler_2.inverse_transform(outputs_fit[699:704]))
    print(' ')
    print('Generated:')
    print(scaler_2.inverse_transform(loaded_model.predict(inputs_fit[699:999], batch_size=10, verbose=0, steps=None)[:5]))
    print('Generated - 200:')
    print(scaler_3.inverse_transform(loaded_model.predict(inputs_testing, batch_size=10, verbose=0, steps=None)))
    
    data = {
        'positionX': '1000',
        'positionY': '500',
        'value': '300',
        'type': 'powerUp_02'
    }
    return jsonify(data)
 
if __name__ == "__main__":
    app.run()