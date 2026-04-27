# CareAI

CareAI is an AI-powered voice companion designed for older adults who live alone and may need emotional reassurance, routine support, and early detection of fragile situations such as loneliness, confusion, missed medication, or physical discomfort.

The project combines a voice interface, a fine-tuned DistilBERT classifier, a local Mistral model through Ollama, short-term emotional memory, and a rule-based alert system. Its goal is not only to answer questions, but to provide human-centered support over time while preserving dignity and autonomy.

## Project Goal

The purpose of CareAI is to support seniors through natural conversation and detect weak signals that may indicate emotional distress or health-related concerns. Instead of reacting only to emergencies, the system is designed to notice repeated patterns such as loneliness, anxiety, or medication issues and alert a caregiver only when necessary.

## Main Features

- Voice-based interaction for accessible communication
- Local LLM response generation using Mistral via Ollama
- Emotion and risk classification using fine-tuned DistilBERT
- Short-term memory to track emotional trends over time
- Rule-based overrides for urgent safety situations
- Selective alerting for family members or nurses

## Target User

CareAI was designed around the persona of an older adult living alone who:
- prefers speaking over typing
- may feel lonely or anxious
- may forget medication
- may struggle with complex digital interfaces
- still wants to remain independent

## System Architecture

The application follows a hybrid architecture:

1. The user speaks to the assistant.
2. Speech is transcribed into text.
3. DistilBERT classifies the emotional or risk-related content.
4. Rule-based safety logic checks for urgent keywords.
5. Mistral generates a short, warm, contextual response.
6. The system updates short-term memory.
7. If needed, an alert is triggered for a caregiver.

## Technologies Used

- Python
- `speech_recognition`
- `pyttsx3`
- Hugging Face Transformers
- DistilBERT
- Ollama
- Mistral
- JSON for labeled training data and memory structure

## Fine-Tuning Approach

The emotion classifier was fine-tuned on a custom JSON dataset containing user sentences and associated labels.

Example:

```json
[
  {"text": "I feel very alone today", "label": "loneliness"},
  {"text": "I forgot my pills this morning", "label": "medication_issue"},
  {"text": "Help, I fell and I feel dizzy", "label": "health_concern"}
]
```

The main labels used in the project are:

- `loneliness`
- `anxiety`
- `health_concern`
- `medication_issue`
- `disorientation`
- `gratitude`
- `neutral`

## Prompt Design

The conversational prompt is structured to keep the assistant supportive, short, and adapted to the user context.

Prompt design includes:
- assistant role definition
- user context
- recent emotional context
- response length constraints
- tone constraints
- a gentle follow-up question

This structure improves consistency, empathy, and safety.

## Safety Logic

Because a care-related system cannot rely only on probabilistic model output, CareAI includes rule-based overrides. If urgent words such as “help”, “pain”, “fall”, or “dizzy” are detected, the system can force a health-related alert even if the classifier is uncertain.

This makes the architecture more interpretable and safer for high-stakes situations.

## Evaluation Scenarios

The prototype can be evaluated with realistic test cases such as:

- “I forgot my pills this morning.”
- “I feel very alone these days.”
- “Help, I fell and I’m dizzy.”
- “I didn’t sleep well, but I’m okay.”

These scenarios help verify:
- label accuracy
- response appropriateness
- memory updates
- alert behavior

## Limitations

This project is a prototype and still has several limitations:

- memory is short-term and session-based
- alerts are simulated rather than connected to real messaging services
- the fine-tuning dataset is limited in size
- the system is not a medical device

## Ethical Considerations

CareAI handles sensitive emotional and potentially health-related information. For this reason, it should be used with:
- explicit user consent
- privacy protection
- transparency about stored data
- clear limits on automation

The assistant is designed to support caregivers, not replace them.

## Future Improvements

Possible next steps include:

- persistent memory across sessions
- secure messaging integration
- multilingual support
- caregiver dashboard
- larger fine-tuning dataset
- more robust evaluation metrics

## Author

**Betül Celikoz**

## Repository Structure

```bash
.
├── data/
├── models/
├── src/
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/your-username/careai.git
cd careai
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Disclaimer

CareAI is an academic prototype created for educational purposes. It is not intended to replace professional medical advice, diagnosis, or emergency services.# CareAI
