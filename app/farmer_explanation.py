import json
import os

class FarmerExplanationEngine:
    def __init__(self, kb_path):
        """
        Initializes the explanation engine and loads the bilingual knowledge base.
        """
        self.kb_path = kb_path
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {}

    def extract_conditions(self, env_data):
        """
        Converts raw environmental numeric parameters into simple language indicators
        for Why the disease propagated.
        """
        reasons_hi = []
        reasons_en = []
        advices_hi = []
        advices_en = []
        
        temp = env_data.get('temp', 25.0)
        hum = env_data.get('hum', 60.0)
        ph = env_data.get('ph', 6.5)
        rain = env_data.get('rain', 100.0)
        n = env_data.get('n', 50.0)
        
        # High Temperature Condition (>35C)
        if temp > 35.0:
            reasons_hi.append("ज्यादा गर्मी है (High Temperature)")
            reasons_en.append("High temperature (>35°C)")
            advices_hi.append("सिंचाई बढ़ाएं और पौधों को छाया प्रदान करें")
            advices_en.append("Increase irrigation to prevent heat stress")
            
        # Cold Stress (<10C)
        if temp < 10.0:
            reasons_hi.append("ठंड बहुत अधिक है (Cold Stress)")
            reasons_en.append("Cold Stress (<10°C)")
            
        # High Humidity (>80%) - Fungal Enabler
        if hum > 80.0:
            reasons_hi.append("हवा में नमी अधिक है (High Humidity)")
            reasons_en.append("High humidity (>80%) provides a fungal breeding ground")
            advices_hi.append("पौधों के बीच हवा का प्रवाह बढ़ाएं (खरपतवार हटाएं)")
            advices_en.append("Clear weeds to maintain airflow between plants")

        # Drought Stress
        if hum < 30.0 and rain < 30.0:
            reasons_hi.append("हवा सूखी है और बारिश की कमी है")
            reasons_en.append("Air is dry with a severe lack of rainfall")
            advices_hi.append("सिंचाई का शेड्यूल कड़ाई से लागू करें")
            advices_en.append("Strictly enforce a heavy irrigation schedule")
            
        # High Rain (Waterlogging)
        if rain > 200.0:
            reasons_hi.append("अत्यधिक बारिश / जलभराव (Excessive Rainfall)")
            reasons_en.append("Heavy rainfall causing waterlogging")
            advices_hi.append("खेत में पानी जमा न होने दें, जल निकासी ठीक करें")
            advices_en.append("Implement proper drainage to prevent root rot")

        # pH Extreme
        if ph < 5.5:
            reasons_hi.append("मिट्टी अत्यधिक अम्लीय है (High Soil Acidity)")
            reasons_en.append("Soil is too acidic (pH < 5.5)")
            advices_hi.append("मिट्टी में चूना (Lime) मिलाएं")
            advices_en.append("Add agricultural lime to balance soil pH")
        elif ph > 8.0:
            reasons_hi.append("मिट्टी क्षारीय है (High Soil Alkalinity)")
            reasons_en.append("Soil is highly alkaline (pH > 8.0)")

        # Nitrogen Toxicity / Vigor
        if n > 120.0:
            reasons_hi.append("नाइट्रोजन अधिक होने से पत्ते कमजोर और कीट-आकर्षक हो गए हैं")
            reasons_en.append("Nitrogen toxicity causes weak, pest-attractive foliage")
            advices_hi.append("उर्वरक (Fertilizer) का प्रयोग कम करें")
            advices_en.append("Reduce nitrogen-heavy fertilizer application immediately")
            
        return reasons_hi, reasons_en, advices_hi, advices_en

    def analyze(self, disease_name, confidence, env_data, stress_score):
        """
        Acts as the intelligent bridge linking the CNN+GNN numeric outputs to Farmer outputs.
        Produces explicit structured bilingual outputs avoiding jargon.
        """
        # Check if the plant is healthy
        is_healthy = "healthy" in disease_name.lower()
        
        if is_healthy:
            # Override KB with healthy specifics
            kb_entry = {
                "Reason_HI": ["पौधा बिल्कुल स्वस्थ दिख रहा है"],
                "Reason_EN": ["The plant appears completely healthy"],
                "Advice_HI": ["नियमित देखभाल जारी रखें"],
                "Advice_EN": ["Continue regular maintenance"]
            }
        else:
            # Fetch KB Data (or fallback to general)
            kb_entry = self.knowledge_base.get(disease_name, self.knowledge_base.get("General", {}))
        
        # 1. Stress Score Interpretation
        if stress_score > 0.6:
            stress_hi = "पौधा बहुत कमजोर है"
            stress_en = "Plant is highly stressed"
        elif stress_score > 0.3:
            stress_hi = "पौधा थोड़ा प्रभावित है"
            stress_en = "Plant is moderately stressed"
        else:
            stress_hi = "पौधा स्वस्थ स्थिति में है"
            stress_en = "Plant is generally healthy"
            
        # 2. Condition Extraction (Merge KB limits with Live Local Environment)
        env_reasons_hi, env_reasons_en, env_advices_hi, env_advices_en = self.extract_conditions(env_data)
        
        final_reasons_hi = env_reasons_hi + kb_entry.get("Reason_HI", [])
        final_reasons_en = env_reasons_en + kb_entry.get("Reason_EN", [])
        final_advices_hi = kb_entry.get("Advice_HI", []) + env_advices_hi
        final_advices_en = kb_entry.get("Advice_EN", []) + env_advices_en

        # 3. Handle Low Confidence (Autonomous pipeline handles the data, we handle the UI UX)
        if confidence < 60.0:
            warning_hi = "हमें पूरी तरह भरोसा नहीं है। कृपया स्पष्ट तस्वीर खींचें या बेहतर प्रकाश में अपलोड करें।"
            warning_en = "Model is not confident. Please upload a clearer image or try a different angle."
            # We degrade the strictness of the advice so farmers don't act on bad predictions
            problem_hi = "अस्पष्ट संक्रमण (संभावित: {})".format(disease_name)
            problem_en = f"Unclear Infection (Possible: {disease_name})"
        else:
            warning_hi = None
            warning_en = None
            if is_healthy:
                problem_hi = "पौधा स्वस्थ है: " + disease_name
                problem_en = "Plant is Healthy: " + disease_name
            else:
                problem_hi = "पत्ते में बीमारी है: " + disease_name
                problem_en = "Leaf is affected by: " + disease_name
            
        # Eliminate duplicates
        final_reasons_hi = list(dict.fromkeys(final_reasons_hi))
        final_reasons_en = list(dict.fromkeys(final_reasons_en))
        final_advices_hi = list(dict.fromkeys(final_advices_hi))
        final_advices_en = list(dict.fromkeys(final_advices_en))

        # Format Final Object
        output = {
            "disease": {
                "hi": problem_hi,
                "en": problem_en
            },
            "stress": {
                "hi": stress_hi,
                "en": stress_en
            },
            "reasons": {
                "hi": final_reasons_hi,
                "en": final_reasons_en
            },
            "advices": {
                "hi": final_advices_hi,
                "en": final_advices_en
            },
            "confidence_warning": {
                "hi": warning_hi,
                "en": warning_en
            }
        }
        
        return output
