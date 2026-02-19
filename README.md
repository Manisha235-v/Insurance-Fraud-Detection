# 🛡️ FraudGuard AI — Insurance Fraud Detection System

A web-based AI-powered system to detect fraudulent health insurance claims in real-time.

---

## 📌 Problem Statement
Insurance fraud costs billions of dollars every year. This project aims to detect fraudulent insurance claims using AI and rule-based logic to help insurance companies identify suspicious claims quickly and accurately.

---

## 💡 Features
- 📝 Claim submission form with multiple fields
- 🔍 Real-time fraud detection and scoring
- ⚠️ Risk level classification (Low / Medium / High)
- 📋 Reasons for fraud detection displayed clearly
- 📸 Accident photo upload support
- 📊 Claims history table
- 🎨 Clean and responsive UI

---

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask (coming soon)
- **AI/ML:** Rule-based scoring (XGBoost integration coming soon)

---

## 📁 Project Structure
```
fraud-detection/
├── frontend/
│   ├── index.html       # Main webpage
│   ├── style.css        # Styling
│   └── script.js        # Fraud detection logic
├── backend/             # Coming soon
│   ├── app.py           # Flask API
│   └── train.py         # ML model training
└── README.md
```

---

## 🚀 How to Run

### Frontend
1. Open the `frontend` folder in VS Code
2. Right click on `index.html`
3. Click **"Open with Live Server"**
4. The app will open in your browser!

### Backend (Coming Soon)
```bash
cd backend
pip install -r requirements.txt
python train.py
python app.py
```

---

## 🔍 How Fraud Detection Works
The system analyzes multiple factors to calculate a **Fraud Score (0-100%)**:

| Factor | Risk Weight |
|--------|------------|
| High claim amount (>$20,000) | +20% |
| No police report filed | +15% |
| Fire/Theft incident | +10-15% |
| No witnesses | +10% |
| Multiple previous claims | +20% |
| No medical report | +15% |
| Low income but high claim | +15% |
| No accident photo | +10% |

**Risk Levels:**
- 🟢 **Low Risk** — Fraud Score below 50%
- 🟠 **Medium Risk** — Fraud Score 50-74%
- 🔴 **High Risk** — Fraud Score 75% and above

---

## 👨‍💻 Built For
This project was built for a **Hackathon** focused on solving real-world problems using AI.

---

## 📃 License
This project is open source and available for educational purposes.
