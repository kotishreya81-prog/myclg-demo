class RiskFusionEngine:
    def __init__(self):
        self.weights = {
            "ecg": 0.20,
            "hr": 0.15,
            "spo2": 0.15,
            "bp": 0.10,
            "rr": 0.10,
            "hrv": 0.10,
            "fatigue": 0.10,
            "stress": 0.05,
            "temp": 0.05
        }
        self.override_threshold = 0.9

    def compute_risk(self, scores):

       
        if scores["ecg"] >= self.override_threshold:
            return 1.0

        risk = 0
        for key in self.weights:
            risk += self.weights[key] * scores.get(key, 0)

        return risk




if __name__ == "__main__":

    fusion = RiskFusionEngine()

    sample_scores = {
        "ecg": 0.3,
        "hr": 0.6,
        "spo2": 0.2,
        "bp": 0.5,
        "rr": 0.4,
        "hrv": 0.3,
        "fatigue": 0.7,
        "stress": 0.5,
        "temp": 0.2
    }

    risk = fusion.compute_risk(sample_scores)

    print("Computed Risk Score:", risk)