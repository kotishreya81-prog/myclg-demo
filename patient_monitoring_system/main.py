class RiskFusionEngine:
    def __init__(self):
       
        self.weights = {
            "ecg": 1.0,
            "fatigue": 1.0,
            "stress": 1.0,
            "bp": 1.0,
            "hr": 1.0,
            "spo2": 1.0,
            "rr": 1.0,
            "hrv": 1.0,
            "temp": 1.0
        }

    def compute(self, scores):
       
        total_weight = sum(self.weights.values())
        risk_sum = sum(scores[k] * self.weights.get(k, 1.0) for k in scores)
        risk = risk_sum / total_weight
        return round(risk, 2) 