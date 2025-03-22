from experta import *

# Heart Disease Expert System using extracted rules
class HeartDiseaseExpert(KnowledgeEngine):
    
    @Rule(Fact(cp=P(lambda x: x > 0.5)), Fact(thal=P(lambda x: x <= 2.5)), Fact(oldpeak=P(lambda x: x > 2.1)), Fact(slope=P(lambda x: x <= 0.5)))
    def high_cp_thal_oldpeak_slope(self):
        self.declare(Fact(risk="high"))
    
    @Rule(Fact(cp=P(lambda x: x <= 0.5)), Fact(ca=P(lambda x: x > 0.5)), Fact(trestbps=P(lambda x: x <= 109)), Fact(chol=P(lambda x: x > 233.5)))
    def high_trestbps_chol(self):
        self.declare(Fact(risk="high"))
    
    @Rule(Fact(exang=P(lambda x: x > 0.5)), Fact(oldpeak=P(lambda x: x > 0.65)), Fact(thal=P(lambda x: x > 2.5)))
    def high_exang_oldpeak_thal(self):
        self.declare(Fact(risk="high"))
    
    @Rule(Fact(age=P(lambda x: x > 55.5)), Fact(oldpeak=P(lambda x: x > 2.1)), Fact(slope=P(lambda x: x <= 0.5)))
    def high_age_oldpeak_slope(self):
        self.declare(Fact(risk="high"))
    
    @Rule(Fact(thal=P(lambda x: x > 2.5)), Fact(thalach=P(lambda x: x <= 132.5)))
    def high_thal_low_thalach(self):
        self.declare(Fact(risk="high"))
    
    @Rule(Fact(risk="high"))
    def alert_high_risk(self):
        print("⚠️ High risk of heart disease! Consult a doctor immediately.")
        self.halt()
    
    @Rule(Fact(cp=P(lambda x: x <= 0.5)), Fact(ca=P(lambda x: x > 0.5)), Fact(trestbps=P(lambda x: x > 109)), Fact(chol=P(lambda x: x <= 233.5)))
    def low_trestbps_chol(self):
        self.declare(Fact(risk="low"))
    
    @Rule(Fact(thalach=P(lambda x: x > 132.5)), Fact(oldpeak=P(lambda x: x <= 1.95)))
    def low_thalach_oldpeak(self):
        self.declare(Fact(risk="low"))
    
    @Rule(Fact(exang=P(lambda x: x <= 0.5)), Fact(oldpeak=P(lambda x: x <= 0.65)), Fact(thal=P(lambda x: x <= 2.5)))
    def low_exang_oldpeak_thal(self):
        self.declare(Fact(risk="low"))
    
    @Rule(Fact(oldpeak=P(lambda x: x <= 2.1)), Fact(slope=P(lambda x: x > 0.5)))
    def low_oldpeak_slope(self):
        self.declare(Fact(risk="low"))
    
    @Rule(Fact(cp=P(lambda x: x > 0.5)), Fact(thal=P(lambda x: x > 2.5)), Fact(thalach=P(lambda x: x > 132.5)))
    def low_cp_thal_thalach(self):
        self.declare(Fact(risk="low"))
    
    @Rule(Fact(risk="low"))
    def alert_low_risk(self):
        print("✅ Low risk of heart disease. Maintain a healthy lifestyle!")

# Function to get user input and run the expert system
def check_heart_disease():
    engine = HeartDiseaseExpert()
    engine.reset()
    
    age = float(input("Enter age: "))
    cp = float(input("Enter chest pain type (0-3): "))
    thalach = float(input("Enter max heart rate achieved (thalach): "))
    exang = float(input("Do you have exercise-induced angina? (0 = No, 1 = Yes): "))
    oldpeak = float(input("Enter ST depression (oldpeak): "))
    ca = float(input("Enter number of major vessels (0-4): "))
    thal = float(input("Enter Thalassemia value (0-3): "))
    slope = float(input("Enter slope of ST segment (0-2): "))
    trestbps = float(input("Enter resting blood pressure: "))
    chol = float(input("Enter cholesterol level: "))
    
    engine.declare(Fact(age=age), Fact(cp=cp), Fact(thalach=thalach), Fact(exang=exang),
                   Fact(oldpeak=oldpeak), Fact(ca=ca), Fact(thal=thal), Fact(slope=slope),
                   Fact(trestbps=trestbps), Fact(chol=chol))
    
    engine.run()

# Run the expert system
if __name__ == "__main__":
    check_heart_disease()
