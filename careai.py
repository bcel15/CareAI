import speech_recognition as sr
import pyttsx3
import time
import requests
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# =========================
# 🔹 CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

labels = [
    "loneliness","social_isolation","sadness","grief","anxiety","fear",
    "confusion","memory_loss","disorientation","frustration","helplessness",
    "boredom","fatigue","sleep_issue","health_concern","medication_issue",
    "appetite_loss","happiness","gratitude","neutral"
]

# =========================
# 🔹 LOAD MODEL
# =========================

tokenizer = DistilBertTokenizer.from_pretrained("careai_model")
model = DistilBertForSequenceClassification.from_pretrained("careai_model")

# =========================
# 🔹 USER SETUP
# =========================

user_name = input("👤 Your name: ")

has_nurse = input("Do you have a nurse? (yes/no): ").lower()
nurse_name = input("Nurse name: ") if has_nurse == "yes" else None

family_name = input("Family contact name: ")
family_relation = input("Relation: ")

# =========================
# 🔹 MEMORY
# =========================

emotional_memory = {
    "recent_emotions": [],
    "risk_score": 0,
    "trend": "stable",
    "last_alert_time": 0
}

conversation_history = []

# =========================
# 🔹 EMOTION DETECTION
# =========================

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    return labels[torch.argmax(logits).item()]

# =========================
# 🔹 OVERRIDE URGENCE
# =========================

def override_emotion(user_input, emotion):
    urgent_words = ["help", "pain", "hurt", "emergency", "can't", "fall", "dizzy"]

    if any(w in user_input.lower() for w in urgent_words):
        return "health_concern"

    return emotion

# =========================
# 🔹 ESCALADE
# =========================

def is_escalating(user_input):
    keywords = [
        "so painful", "very painful", "help me", "can't move",
        "worse", "getting worse", "really bad", "too much pain"
    ]
    return any(k in user_input.lower() for k in keywords)

# =========================
# 🔹 MEMORY UPDATE
# =========================

def update_memory(emotion):
    memory = emotional_memory

    memory["recent_emotions"].append(emotion)
    if len(memory["recent_emotions"]) > 5:
        memory["recent_emotions"].pop(0)

    risk_map = {
        "neutral": 0, "happiness": 0, "gratitude": 0,
        "boredom": 1, "fatigue": 1,
        "sadness": 2, "loneliness": 3,
        "anxiety": 3, "fear": 4,
        "helplessness": 5,
        "health_concern": 8, "disorientation": 6,
        "medication_issue": 5
    }

    memory["risk_score"] += risk_map.get(emotion, 1)

    if memory["risk_score"] > 25:
        memory["trend"] = "critical"
    elif memory["risk_score"] > 12:
        memory["trend"] = "declining"
    else:
        memory["trend"] = "stable"

# =========================
# 🔹 ALERT SYSTEM
# =========================

def should_alert(user_input, emotion):
    memory = emotional_memory
    alerts = []

    # 🚨 médical direct
    if emotion == "health_concern":
        alerts += ["family"]
        if has_nurse == "yes":
            alerts += ["nurse"]

    # 🚨 médicaments
    if emotion == "medication_issue" and has_nurse == "yes":
        alerts.append("nurse")

    # 🚨 solitude répétée
    if memory["recent_emotions"].count("loneliness") >= 3:
        alerts.append("family")

    # 🚨 escalade
    if is_escalating(user_input):
        alerts += ["family"]
        if has_nurse == "yes":
            alerts += ["nurse"]

    # 🚨 tendance critique
    if memory["trend"] == "critical":
        alerts += ["family"]
        if has_nurse == "yes":
            alerts += ["nurse"]

    return list(set(alerts))

# =========================
# 🔹 ALERT MESSAGE
# =========================

def build_alert(name, emotion):

    if emotion == "loneliness":
        return f"{name} is feeling very lonely and would really appreciate a call."

    if emotion == "health_concern":
        return f"{name} may not be feeling well physically. Please check on them as soon as possible."

    if emotion == "medication_issue":
        return f"{name} may have missed their medication. Please check on them."

    return f"{name} may need support right now."

# =========================
# 🔹 OLLAMA CALL
# =========================

def ask_mistral(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 120,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["\n\n", "User:", "AI:"]
            }
        }
    )

    data = response.json()
    text = data.get("response", "").strip()

    # FIX phrase coupée
    if text and text[-1] not in [".", "!", "?"]:
        text += "..."

    return text

# =========================
# 🔹 RESPONSE GENERATION
# =========================

def generate_response(user_input):

    emotion = predict_emotion(user_input)
    emotion = override_emotion(user_input, emotion)

    update_memory(emotion)
    alerts = should_alert(user_input, emotion)

    # Ajoute le message à l'historique
    conversation_history.append(f"User: {user_input}")

    # Garde les 10 derniers échanges pour le contexte
    history_text = "\n".join(conversation_history[-10:])

    prompt = f"""
You are a caring companion for an elderly person named {user_name}.

Rules:
- Remember everything the user tells you (name, preferences, stories)
- If the user corrects their name, use the new one from that point on
- Speak naturally like a human
- Keep responses short (2-3 sentences max)
- Always finish your sentences
- Never cut your response
- Be warm and supportive

Conversation so far:
{history_text}

Emotion detected: {emotion}

Response:
"""

    response = ask_mistral(prompt)

    # Ajoute la réponse à l'historique
    conversation_history.append(f"AI: {response}")

    return response, emotion, alerts

# =========================
# 🔹 SPEECH RECOGNITION
# =========================

def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("🎤 Speak...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)

        text = recognizer.recognize_google(audio)
        print("You:", text)

        time.sleep(0.3)

        return text

    except Exception as e:
        print("❌ Error:", e)
        return ""

# =========================
# 🔹 TTS
# =========================

def speak(text):
    clean_text = text.replace("\n", " ").strip()
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        for voice in voices:
            if "en" in voice.languages or "english" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("❌ TTS error:", e)

# =========================
# 🔹 CHAT LOOP
# =========================

print("\n💬 CareAI ready. Type 'quit' to exit.\n")

while True:
    user_input = listen()

    if user_input == "":
        continue

    if user_input.lower() == "quit":
        break

    response, emotion, alerts = generate_response(user_input)

    print(f"\nAI: {response}")
    print(f"(emotion: {emotion})")
    speak(response)

    if alerts:
        msg = build_alert(user_name, emotion)

        for a in alerts:
            if a == "family":
                print(f"📩 Alert sent to {family_relation} ({family_name}): {msg}")

            if a == "nurse" and has_nurse == "yes":
                print(f"📩 Alert sent to nurse ({nurse_name}): {msg}")

    print()