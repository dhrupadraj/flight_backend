from model.model_loader import load_predrnn_model


def predict_wind(wind_history):
    model = load_predrnn_model()
    return model.predict(wind_history)
