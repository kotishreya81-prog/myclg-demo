import joblib


class StressModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        return self.model.predict_proba([features])[0][1]


# 👇 This must come AFTER the class
if __name__ == "__main__":
    import numpy as np

    model = StressModel("stress_model.pkl")

    sample = np.random.rand(9)
    prob = model.predict(sample)

    print("Stress Probability:", prob)
    