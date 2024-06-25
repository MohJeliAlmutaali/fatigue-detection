import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    cred = credentials.Certificate("./sensor-data-aac51-firebase-adminsdk-c1c5b-13c2b4dcf6.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sensor-data-aac51-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
