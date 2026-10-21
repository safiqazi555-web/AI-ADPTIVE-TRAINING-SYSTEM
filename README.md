# AI-Based Adaptive Training System using Artificial Neural Networks

A fully Python project: **Streamlit** web interface, **SQLite** database, **TensorFlow/Keras** ANN,
**scikit-learn** baselines. It follows the final project report (BS(AI)-5C, SZABIST Islamabad) section by section.

## 1. Run it in VS Code (Windows)

1. Install **Python 3.10, 3.11 or 3.12** (recommended for TensorFlow) from python.org. Tick "Add Python to PATH".
2. In VS Code: **File > Open Folder...** and choose this `adaptive_training_system` folder.
3. Open the terminal (**Terminal > New Terminal**) and run:

```bash
python -m venv venv
venv\Scripts\activate                # (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_all.py                  # creates data, trains the ANN, evaluates it (about 1 minute)
streamlit run app.py                 # then open http://localhost:8501 in the browser
```

In VS Code choose the interpreter of the `venv` (Ctrl+Shift+P > "Python: Select Interpreter").
If you skip `python train_all.py`, the website shows a **Train model now** button that does the same.

> If TensorFlow does not install on your PC, delete the `tensorflow` line from `requirements.txt`
> and install again. The ANN then runs with an equivalent scikit-learn MLP (same 64-32-3 layers),
> and everything else works the same.

## 2. Login accounts

| Role | Email | Password |
|---|---|---|
| Admin | `admin@demo.com` | `admin123` |
| Trainee (weak demo) | `ali@demo.com` | `demo123` |
| Trainee (average demo) | `sara@demo.com` | `demo123` |
| Trainee (strong demo) | `hamza@demo.com` | `demo123` |

You can also register a new trainee on the login page. To reset all users and quiz history,
delete `data/training_system.db` and restart the app.

## 3. What the website does

**Trainee**
* **Dashboard** - predicted level, average score, topics completed, charts of scores per topic and score progress, next recommended lesson.
* **Learn & Quiz** - 4 topics (Python Basics, Control Flow, Functions, Data Structures) x 3 difficulties (Easy/Medium/Hard), 4 questions each. Score, time and attempts are saved; the level is predicted again after every quiz.
* **Performance Prediction** - sliders for the 7 input features; the ANN shows level, probabilities and normalised input factors (what-if tool).
* **Early Warning** - risk gauge (Low/Medium/High), study suggestions, and the adaptive learning path.

**Admin**
* **Overview** - all trainees with predicted level, level distribution chart.
* **Trainee Detail** - charts, features and quiz history of one trainee.
* **Model Evaluation** - comparison table (Table 8 of the report), confusion matrices, training curves, retrain button.

## 4. Files and the report

| File | Purpose (report Table 5) |
|---|---|
| `generate_dataset.py` | Synthetic dataset (4,000 trainees, 7 features, 3 classes) |
| `data_preprocessing.py` | Clean, scale, one-hot, 70/15/15 stratified split |
| `model.py` | ANN 7-64-32-3 (ReLU, Dropout 0.3/0.2, Softmax, Adam 0.001, Early Stopping) |
| `evaluate.py` | Accuracy, precision, recall, F1, confusion matrix, comparison with Logistic Regression and Decision Tree (entropy) |
| `adaptive_engine.py` | Predicted level -> next lesson and difficulty (Table 4), risk level, suggestions |
| `database.py` | SQLite: users, lessons, quiz_results, progress (+ login_log for login activity) |
| `app.py` | Streamlit interface: login, quiz, dashboard, admin |
| `predictor.py`, `charts.py`, `lesson_content.py`, `config.py`, `train_all.py` | Glue code, charts, lesson texts and questions, settings, one-command training |

Output files after training: `models/` (ANN, scaler), `reports/` (`model_comparison.csv`, confusion matrices, `training_curves.png`).
Copy your own numbers from `reports/model_comparison.csv` into **Table 8** of the report. They can differ slightly from run to run and PC to PC.

## 5. How the live prediction works

After each quiz the app computes the 7 features from the database:
`avg_quiz_score`, `last_quiz_score`, `time_spent` (minutes), `attempts_per_quiz`, `lessons_completed` (% of topics passed with 60%+),
`login_frequency` (logins per week, last 4 weeks), `days_since_last_login`. They are scaled with the training scaler
(values outside the training range are clipped) and fed to the ANN, which outputs Weak / Average / Strong.
The adaptive engine then picks the next lesson: Weak -> Easy and revision, Average -> Medium, Strong -> Hard;
a very low score in one topic sends the trainee back to that topic only.

## 6. Notes

* The dataset is **synthetic** (generated with realistic rules plus noise), as allowed in Section 5.2 of the report.
  To use a real dataset, put a CSV with the same 8 columns in `data/trainee_dataset.csv` and rerun `python train_all.py`.
* Predictions are estimates to support learning, not official grades.
