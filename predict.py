import pickle
import numpy as np
import tensorflow as tf

# Reset TF graph state for compatibility mode (prevents tf.reset_default_graph deprecation warnings in mixed 1.x/2.x code paths)
tf.compat.v1.reset_default_graph()

rf = pickle.load(open("models/rf_model.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))
encoder = pickle.load(open("models/encoder.pkl","rb"))
dl_model = tf.keras.models.load_model("models/dl_model.h5")

def predict_movie(budget,genre,rating,cast):

    # for unknown genre, OneHotEncoder(handle_unknown='ignore') returns all-zero row
    genre_encoded = encoder.transform([[genre]]).toarray()

    features = np.hstack(([budget,rating,cast],genre_encoded[0]))
    features = scaler.transform([features])

    rf_pred = rf.predict(features)[0]

    dl_pred = dl_model.predict(features)[0][0]

    return rf_pred, dl_pred