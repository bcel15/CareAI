import speech_recognition as sr
import pyttsx3
import time
import requests
import torch
import json
import os
import yaml
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# =========================
# 🔹 CONFIG  
# =========================

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg.setdefault("risk_map", {})
    cfg.setdefault("generation", {
        "num_predict": 200,
        "temperature": 0.7,
        "top_p": 0.9
    })
    cfg.setdefault("risk_thresholds", {
        "decay_minutes": 10,
        "critical": 10,
        "declining": 5
    })
OLLAMA_URL  = cfg["ollama_url"]
MODEL_NAME  = cfg["model_name"]
MEMORY_FILE = cfg["memory_file"]

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
# 🔹 PERSISTENCE  
# =========================

def load_memory():
    """Load saved memory from disk, or return defaults if none exists."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            print("💾 Memory loaded from previous session.")
            return data
        except Exception:
            pass
    return None

def save_memory():
    """Persist emotional memory, conversation history and user info to disk."""
    data = {
        "user_name": user_name,
        "has_nurse": has_nurse,
        "nurse_name": nurse_name,
        "family_name": family_name,
        "family_relation": family_relation,
        "emotional_memory": {
            k: v for k, v in emotional_memory.items()
            if k != "last_decay_time"  
        },
        "conversation_history": conversation_history[-20:] 
    }
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =========================
# 🔹 USER SETUP
# =========================

saved = load_memory()

if saved:
    user_name      = saved["user_name"]
    has_nurse      = saved["has_nurse"]
    nurse_name     = saved["nurse_name"]
    family_name    = saved["family_name"]
    family_relation = saved["family_relation"]
    print(f"👤 Welcome back, {user_name}!")
else:
    user_name = input("👤 Your name: ")
    has_nurse = input("Do you have a nurse? (yes/no): ").lower()
    nurse_name = input("Nurse name: ") if has_nurse == "yes" else None
    family_name = input("Family contact name: ")
    family_relation = input("Relation: ")

# =========================
# 🔹 MEMORY
# =========================

if saved and "emotional_memory" in saved:
    emotional_memory = saved["emotional_memory"]
    emotional_memory["last_decay_time"] = time.time() 
else:
    emotional_memory = {
        "recent_emotions": [],
        "risk_score": 0,
        "trend": "stable",
        "last_alert_time": 0,
        "last_decay_time": time.time()
    }

if saved and "conversation_history" in saved:
    conversation_history = saved["conversation_history"]
else:
    conversation_history = []

# =========================
# 🔹 TTS ENGINE (singleton) 
# =========================

def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    for voice in voices:
        if "en" in voice.languages or "english" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    return engine

tts_engine = init_tts_engine()

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

    memory["risk_score"] += cfg["risk_map"].get(emotion, 1)

    now = time.time()
    minutes_elapsed = (now - memory["last_decay_time"]) / 60
    decay_minutes = cfg["risk_thresholds"].get("decay_minutes", 10)

    if decay_minutes > 0:
        decay_amount = int(minutes_elapsed / decay_minutes)
    else:
        decay_amount = 0

    if decay_amount > 0:
        memory["risk_score"] = max(0, memory["risk_score"] - decay_amount)
        memory["last_decay_time"] = now

    if memory["risk_score"] > cfg["risk_thresholds"]["critical"]:
        memory["trend"] = "critical"
    elif memory["risk_score"] > cfg["risk_thresholds"]["declining"]:
        memory["trend"] = "declining"
    else:
        memory["trend"] = "stable"

# =========================
# 🔹 ALERT SYSTEM
# =========================

def should_alert(user_input, emotion):
    memory = emotional_memory
    alerts = []

    if emotion == "health_concern":
        alerts += ["family"]
        if has_nurse == "yes":
            alerts += ["nurse"]

    if emotion == "medication_issue" and has_nurse == "yes":
        alerts.append("nurse")

    if memory["recent_emotions"].count("loneliness") >= 3:
        alerts.append("family")

    if is_escalating(user_input):
        alerts += ["family"]
        if has_nurse == "yes":
            alerts += ["nurse"]

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
                "num_predict": cfg["generation"]["num_predict"],
                "temperature": cfg["generation"]["temperature"],
                "top_p":       cfg["generation"]["top_p"],
                "stop": ["\n\n", "User:", "AI:"]
            }
        }, timeout=10
    )

    if response.status_code != 200:
        return "Error"

    data = response.json()
    text = data.get("response", "").strip()

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

    conversation_history.append(f"User: {user_input}")
    if len(conversation_history) > 50:
        conversation_history[:] = conversation_history[-50:]
  
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

Emotional context:
- Current emotion: {emotion}
- Emotional trend: {emotional_memory['trend']} (risk score: {emotional_memory['risk_score']})
- Recent emotions: {', '.join(emotional_memory['recent_emotions']) if emotional_memory['recent_emotions'] else 'none'}

Conversation so far:
{history_text}

Response:
"""

    response = ask_mistral(prompt)

    conversation_history.append(f"AI: {response}")

    save_memory()

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
        tts_engine.say(clean_text)        
        tts_engine.runAndWait()
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