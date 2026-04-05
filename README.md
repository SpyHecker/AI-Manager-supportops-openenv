# 🧠 AI Manager for Customer Support (OpenEnv)

An AI-powered **customer support management environment** built using the OpenEnv framework.  
This project simulates real-world customer support workflows where an AI agent processes tickets through classification, prioritization, routing, and resolution.

---

## 🚀 Overview

This environment enables training and evaluation of AI agents on realistic support operations such as:

- Ticket classification
- Priority assignment
- Team routing
- Customer interaction
- Escalation handling
- Resolution workflows

The system follows the **OpenEnv standard** with `step()`, `reset()`, and `state()` APIs, making it fully compatible with agent-based evaluation pipelines.

---

## 🎯 Motivation

Customer support is a critical real-world task involving:

- Multi-step decision making
- Context understanding
- Risk handling (security issues)
- Communication skills

This environment provides a **structured simulation** to train AI agents for such tasks.

---

## 🏗️ Environment Design

### Core API

- `reset()` → Initializes a new support ticket
- `step(action)` → Executes an action and returns:
  - observation
  - reward
  - done
  - info
- `state()` → Returns current internal state

---

## 📦 Action Space

```json
{
  "action_type": "string",
  "category": "string (optional)",
  "priority": "string (optional)",
  "team": "string (optional)",
  "message": "string (optional)",
  "escalation_reason": "string (optional)"
}
```
---

### 🎬 Valid Actions

- `classify_ticket`
- `set_priority`
- `assign_team`
- `request_more_info`
- `draft_response`
- `escalate_ticket`
- `resolve_ticket`
- `close_without_resolution`

---

## 👀 Observation Space

Each observation includes:

- `customer_message`
- `current_category`
- `current_priority`
- `assigned_team`
- `ticket_history`
- `flags` (e.g., security risk, urgency)
- `ready_for_resolution`

---

## 🧪 Tasks

### 1. Easy Task — Billing Issue

- **Scenario:** Double charge / refund request
- **Goal:**
  - classify → `billing`
  - assign → `billing_team`
  - resolve ticket
- **Difficulty:** ⭐

### 2. Medium Task — OTP / Login Issue

- **Scenario:** OTP not received before urgent event
- **Goal:**
  - classify → `account_access`
  - assign → `account_recovery`
  - request_more_info or assist user
- **Difficulty:** ⭐⭐

### 3. Hard Task — Security + Billing Conflict

- **Scenario:** Refund request + suspicious login activity
- **Goal:**
  - Prioritize security
  - Escalate ticket
- **Difficulty:** ⭐⭐⭐

---

## 🏆 Reward Function

Rewards are dense and step-based, not binary:

| Action                  | Outcome          |
|-------------------------|------------------|
| Correct classification  | Positive reward  |
| Correct priority        | Positive reward  |
| Correct team            | Positive reward  |
| Good response           | Small reward     |
| Correct escalation      | High reward      |
| Wrong actions           | Penalties        |

---

## 🤖 Baseline Agent

- Uses OpenAI-compatible API
- Structured JSON output
- Prompt-constrained decisions
- Context-aware fallback logic

### Baseline Scores

| Task   | Score |
|--------|-------|
| Easy   | 1.00  |
| Medium | 0.70  |
| Hard   | 0.75  |

---

## 🛠️ Setup Instructions

### 1. Clone repo

```bash
git clone <your-repo-url>
cd supportops-openenv
```

### 2. Create virtual environment

```bash
py -3.10 -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```env
API_BASE_URL=your_api_endpoint
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_api_key
```

### 5. Run server

```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

### 6. Run inference

```bash
python inference.py
```

---

## 🐳 Docker

### Build

```bash
docker build -t supportops-env .
```

### Run

```bash
docker run -p 7860:7860 supportops-env
```

---

## 🌐 Deployment (Hugging Face Spaces)

```bash
openenv push --repo-id your-username/supportops-openenv
```

---

## ✅ OpenEnv Compliance

- ✔ Typed models (Pydantic)
- ✔ `step` / `reset` / `state` APIs
- ✔ `openenv.yaml`
- ✔ 3 tasks with graders
- ✔ Reward shaping
- ✔ Baseline inference script
- ✔ Docker support

---

## 🧠 Key Features

- Real-world simulation (customer support)
- Multi-step reasoning environment
- Security-first handling logic
- Context-aware fallback system
- Structured evaluation pipeline

---

## 📂 Project Structure

```

supportops-openenv/
|
├──app/
|   ├── server.py
|   ├── environment.py
|   ├── models.py
|   |── utils.py
|   |── rewards.py
|   |── graders.py
|   └── tasks.py
|
├──data/
|   ├── easy_task.json
|   ├── medium_task.json
|   └── hard_task.json
|
├──tests/
|   ├── test_env.py
|   ├── test_graders.py
|   ├── test_tasks.py
|
├──.gitignore 
├──inference.py
├──openenv.yaml
├──requirements.txt
├──Dockerfile
├──README.md
```

---

## ⚠️ Constraints

- Runtime < 20 minutes
- Works on 2 vCPU / 8GB RAM
- Deterministic graders
- Reproducible baseline

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Developed for **OpenEnv Hackathon** — AI Manager, Customer Support Domain

---

## 🚀 Future Improvements

- Multi-agent coordination
- Real-time chat simulation
- RL-based training integration
- More complex edge-case scenarios