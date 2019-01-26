from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import csv
from flask import Flask, jsonify, request

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

numPlayers = 0
startLvl = False

playerOne_posX = 0
playerOne_posY = 0
playerTwo_posX = 0
playerTwo_posY = 0
playerThree_posX = 0
playerThree_posY = 0
playerFour_posX = 0
playerFour_posY = 0
 
@app.route("/", methods=['GET', 'PUT', 'POST'])
def predict():

    print(request.content_type)
    
    prediction = scaler_3.inverse_transform(loaded_model.predict(inputs_testing, batch_size=10, verbose=0, steps=None))

    if(request.get_json()):
        rq_body = request.get_json()
        print(rq_body)
        print("health: " + str(rq_body['health']))
        
    data = {
        'positionX': '1000',
        'positionY': '500',
        'value': '300',
        'type': 'powerUp_02'
    }
    print(prediction)
    return jsonify(data)

@app.route("/start", methods=['POST'])
def start():
    global numPlayers
    global startLvl

    if(request.get_json()['ready'] == 'False'):
        if(numPlayers < 4):
            newPlayerID = numPlayers + 1
            numPlayers = newPlayerID
            data = {
                'player_num': newPlayerID,
                'start': 'False'
            }
        else:
            data = {
                'start': 'True'
            }
            startLvl = True
    else:
        if(numPlayers == 4):
            data = {
                'start': 'True'
            }
        else:
            data = {
                'start': 'False'
            }
    print(numPlayers)
    print(startLvl)
    return jsonify(data)

@app.route("/update", methods=['POST'])
def update():
    global playerOne_posX
    global playerOne_posY
    global playerTwo_posX
    global playerTwo_posY
    global playerThree_posX
    global playerThree_posY
    global playerFour_posX
    global playerFour_posY
    
    if(request.get_json()['playerID'] == '1'):
        playerOne_posX = request.get_json()['posX']
        playerOne_posY = request.get_json()['posY']
    elif(request.get_json()['playerID'] == '2'):
        playerTwo_posX = request.get_json()['posX']
        playerTwo_posY = request.get_json()['posY']
    elif(request.get_json()['playerID'] == '3'):
        playerThree_posX = request.get_json()['posX']
        playerThree_posY = request.get_json()['posY']
    elif(request.get_json()['playerID'] == '4'):
        playerFour_posX = request.get_json()['posX']
        playerFour_posY = request.get_json()['posY']        
    
    data = {
            'playerOne_posX': playerOne_posX,
            'playerOne_posY': playerOne_posY,
            'playerTwo_posX': playerTwo_posX,
            'playerTwo_posY': playerTwo_posY,
            'playerThree_posX': playerThree_posX,
            'playerThree_posY': playerThree_posY,
            'playerFour_posX': playerFour_posX,
            'playerFour_posY': playerFour_posY,
        }
    return jsonify(data)
 
if __name__ == "__main__":
    app.run()