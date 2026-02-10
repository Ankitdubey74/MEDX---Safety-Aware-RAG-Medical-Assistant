import re
from typing import Dict, Any, List
import numpy as np

def symptom_risk_score(symptoms: str) -> Dict[str, Any]:
    """
    Calculate risk score based on symptoms.
    Input: "chest pain, shortness of breath, nausea"
    """
    # Common high-risk symptoms
    high_risk = ["chest pain", "shortness of breath", "pressure chest", "nausea", "sweating"]
    medium_risk = ["fatigue", "dizziness", "palpitations", "arm pain"]
    
    symptom_list = re.split(r'[,;]', symptoms.lower())
    symptom_list = [s.strip() for s in symptom_list if s.strip()]
    
    high_count = sum(1 for s in symptom_list if any(hr in s for hr in high_risk))
    medium_count = sum(1 for s in symptom_list if any(mr in s for mr in medium_risk))
    
    # Risk scoring logic
    if high_count >= 2:
        risk_score = 0.85
        level = "🚨 EMERGENCY"
    elif high_count == 1:
        risk_score = 0.65
        level = "⚠️ HIGH"
    elif medium_count >= 2:
        risk_score = 0.45
        level = "🟡 MODERATE"
    else:
        risk_score = 0.15
        level = "🟢 LOW"
    
    return {
        "risk_score": risk_score,
        "risk_level": level,
        "high_risk_symptoms_found": high_count,
        "medium_risk_symptoms_found": medium_count,
        "recommendation": "HIGH/EMERGENCY → Seek immediate medical help" if risk_score > 0.6 else "Consult doctor"
    }

def bmi_calculator(weight_kg: float, height_cm: float) -> Dict[str, Any]:
    """
    BMI calculator with category
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
        risk = "Low-Moderate"
    elif 18.5 <= bmi < 25:
        category = "Normal"
        risk = "Low"
    elif 25 <= bmi < 30:
        category = "Overweight"
        risk = "Moderate"
    else:
        category = "Obese"
        risk = "High"
    
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "health_risk": risk,
        "advice": f"{'Maintain' if risk=='Low' else 'Consult nutritionist for'} weight management"
    }

def emergency_flag(symptoms: str, vitals: str = "") -> Dict[str, Any]:
    """
    Check for emergency symptoms
    """
    emergency_keywords = [
        "chest pain", "cannot breathe", "unconscious", "seizure", 
        "severe bleeding", "stroke symptoms", "heart attack"
    ]
    
    symptom_lower = symptoms.lower()
    is_emergency = any(keyword in symptom_lower for keyword in emergency_keywords)
    
    vitals_risk = 0
    if vitals:
        vitals_lower = vitals.lower()
        vitals_risk_keywords = ["bp >180", "heart rate >200", "oxygen <90"]
        vitals_risk = sum(1 for k in vitals_risk_keywords if k in vitals_lower)
    
    return {
        "is_emergency": is_emergency or vitals_risk > 0,
        "emergency_score": 0.9 if is_emergency else 0.3 + (vitals_risk * 0.2),
        "action": "🚑 CALL EMERGENCY NOW!" if is_emergency else "👨‍⚕️ Consult doctor soon",
        "triggers": [k for k in emergency_keywords if k in symptom_lower]
    }

# Tool descriptions for LLM
TOOLS = {
    "symptom_risk_score": {
        "description": "Calculate risk score from patient symptoms. Use when user mentions specific symptoms like chest pain, shortness of breath.",
        "parameters": {"symptoms": "string - comma separated symptoms"},
        "when_to_use": "numerical risk assessment needed"
    },
    "bmi_calculator": {
        "description": "Calculate BMI and health category from weight and height.",
        "parameters": {"weight_kg": "float", "height_cm": "float"},
        "when_to_use": "user asks about weight, BMI, obesity"
    },
    "emergency_flag": {
        "description": "Check if symptoms indicate medical emergency.",
        "parameters": {"symptoms": "string", "vitals": "optional string"},
        "when_to_use": "urgent symptoms mentioned"
    }
}
