

class DroneStateMachine:
    def __init__(self, threshold=0.85):
        self.state = "IDLE"
        self.threshold = threshold
        self.timer = 0

    def update(self, risk):

        print(f"Current State: {self.state} | Risk: {risk:.2f}")

        if self.state == "IDLE":
            if risk >= self.threshold:
                print("⚠ Risk Detected!")
                self.state = "RISK_DETECTED"
                self.timer = 0

        elif self.state == "RISK_DETECTED":
            self.timer += 1
            print(f"Confirming... ({self.timer}/5)")

            if self.timer >= 5:
                self.state = "ARMED"

        elif self.state == "ARMED":
            self.deploy_drone()
            self.state = "DEPLOYED"

        elif self.state == "DEPLOYED":
            print("Drone already deployed.")

    def deploy_drone(self):
        print("🚁 Drone Deployment Triggered")



if __name__ == "__main__":

    import time

    drone = DroneStateMachine(threshold=0.85)

   
    simulated_risk_values = [0.2, 0.4, 0.9, 0.92, 0.95, 0.97, 0.99, 0.5]

    for risk in simulated_risk_values:
        drone.update(risk)
        time.sleep(1)