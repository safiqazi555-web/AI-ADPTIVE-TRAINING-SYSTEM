"""Streamlit interface of the AI-Based Adaptive Training System.

Run:  streamlit run app.py
Trainee pages : Dashboard | Learn & Quiz | Performance Prediction | Early Warning
Admin pages   : Overview | Trainee Detail | Model Evaluation
"""
import random
import time

import pandas as pd
import streamlit as st

import charts
import database as db
from config import (ADMIN_EMAIL, ADMIN_PASSWORD, DEMO_PASSWORD, DIFFICULTIES, FEATURE_LABELS,
                    FEATURES, LEVELS, MODEL_DIR, PASS_MARK, REPORT_DIR)
from lesson_content import LESSONS, TOPICS

st.set_page_config(page_title="Adaptive Training System", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.badge {display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; color:#10131f;}
.card {border:1px solid rgba(255,255,255,0.10); border-radius:14px; padding:16px 18px;
       background:rgba(255,255,255,0.03); margin-bottom:12px;}
.card h4 {margin:0 0 6px 0;}
.muted {color:rgba(232,235,245,0.65); font-size:0.92rem;}
.warn {border-left:4px solid #EF6B6B;} .good {border-left:4px solid #46D2A0;} .tip {border-left:4px solid #F2B84B;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Start-up
# --------------------------------------------------------------------------- #
@st.cache_resource
def _startup():
    db.init_db()
    return True


_startup()


def model_ready() -> bool:
    return (MODEL_DIR / "model_info.json").exists()


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def analyse(user_id: int) -> dict:
    import predictor
    return predictor.analyse_user(user_id)


def train_model_ui():
    with st.spinner("Creating the dataset, training the ANN and evaluating it (about 1 minute)..."):
        from config import DATASET_PATH
        if not DATASET_PATH.exists():
            from generate_dataset import generate
            DATASET_PATH.parent.mkdir(exist_ok=True)
            generate().to_csv(DATASET_PATH, index=False)
        import evaluate
        import model
        import predictor
        model.train(verbose=False)
        evaluate.evaluate()
        predictor._cache.clear()
    st.success("Model trained.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Login / registration
# --------------------------------------------------------------------------- #
def login_page():
    st.title("🎓 AI-Based Adaptive Training System")
    st.caption("A neural network estimates your level and adapts the next lesson to you.")
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        tab_in, tab_reg = st.tabs(["Log in", "Register"])
        with tab_in:
            with st.form("login"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in"):
                    user = db.authenticate(email, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Wrong email or password.")
        with tab_reg:
            with st.form("register"):
                name = st.text_input("Full name")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password (min 4 characters)", type="password", key="reg_pw")
                if st.form_submit_button("Create trainee account"):
                    ok, msg = db.create_user(name, email, password, "trainee")
                    st.success("Account created. You can log in now.") if ok else st.error(msg)
        with st.expander("Demo accounts"):
            st.markdown(f"**Admin:** `{ADMIN_EMAIL}` / `{ADMIN_PASSWORD}`  \n"
                        f"**Trainees:** `ali@demo.com` (weak), `sara@demo.com` (average), "
                        f"`hamza@demo.com` (strong) / `{DEMO_PASSWORD}`")


# --------------------------------------------------------------------------- #
# Trainee pages
# --------------------------------------------------------------------------- #
def go_learn(topic: str, difficulty: str):
    """Button callback: jump to the lesson page with a lesson pre-selected."""
    st.session_state["nav_trainee"] = "Learn & Quiz"
    st.session_state["sel_topic"] = topic
    st.session_state["sel_diff"] = difficulty


def recommendation_card(rec: dict, key: str):
    st.markdown(f'<div class="card"><h4>Next step: {rec["action"]} · {rec["topic"]} · {rec["difficulty"]}</h4>'
                f'<div class="muted">{rec["message"]}</div></div>', unsafe_allow_html=True)
    st.button(f"Open {rec['topic']} ({rec['difficulty']})", key=key, on_click=go_learn,
              args=(rec["topic"], rec["difficulty"]))


def page_dashboard(user):
    a = analyse(user["user_id"])
    st.header("Student Dashboard")
    st.caption(f"Welcome, {user['name']}. Your marks, activity and progress at a glance.")
    results, pred, feats, rec = a["results"], a["prediction"], a["features"], a["recommendation"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Predicted level**")
        if pred:
            st.markdown(badge(pred["level"], charts.LEVEL_COLORS[pred["level"]]) +
                        f' <span class="muted">{pred["confidence"] * 100:.0f}% sure</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="muted">Take a quiz first</span>', unsafe_allow_html=True)
    c2.metric("Average score", f"{feats['avg_quiz_score']:.0f}%" if feats else "-")
    c3.metric("Topics completed", f"{feats['lessons_completed']:.0f}%" if feats else "0%")
    c4.metric("Quizzes taken", len(results))

    if results.empty:
        st.info("No quiz results yet. Start with the placement lesson below.")
        recommendation_card(rec, "dash_start")
        return

    left, right = st.columns(2)
    left.plotly_chart(charts.topic_scores(results))
    right.plotly_chart(charts.score_progress(results))

    recommendation_card(rec, "dash_next")
    st.subheader("Recent quiz results")
    recent = results.tail(8).iloc[::-1][["timestamp", "topic", "difficulty", "score", "attempts", "time_spent"]]
    recent.columns = ["When", "Topic", "Difficulty", "Score (%)", "Attempt no.", "Minutes"]
    st.dataframe(recent, hide_index=True)


def page_learn(user):
    a = analyse(user["user_id"])
    rec = a["recommendation"]
    st.header("Learn & Quiz")
    st.session_state.setdefault("sel_topic", rec["topic"])
    st.session_state.setdefault("sel_diff", rec["difficulty"])

    c1, c2, c3 = st.columns([2, 1, 2])
    topic = c1.selectbox("Topic", TOPICS, key="sel_topic")
    diff = c2.selectbox("Difficulty", DIFFICULTIES, key="sel_diff")
    with c3:
        st.write("")
        if (topic, diff) == (rec["topic"], rec["difficulty"]):
            st.markdown(badge("Recommended for you", "#7C9CFF"), unsafe_allow_html=True)
        else:
            st.caption(f"Recommended: {rec['topic']} ({rec['difficulty']})")

    lesson = LESSONS[(topic, diff)]
    with st.container(border=True):
        st.subheader(f"{topic} · {diff}")
        st.markdown(lesson["content"])

    # timer starts when the lesson is opened (reset after every submitted quiz)
    timers = st.session_state.setdefault("lesson_timers", {})
    timers.setdefault((topic, diff), time.time())

    lid = db.lesson_id(topic, diff)
    results = a["results"]
    attempt_no = int((results[(results["topic"] == topic) & (results["difficulty"] == diff)]).shape[0])

    st.subheader("Quiz")
    with st.form(f"quiz_{lid}_{attempt_no}"):
        answers = []
        for i, q in enumerate(lesson["questions"]):
            st.markdown(f"**Q{i + 1}. {q['q']}**")
            if q["code"]:
                st.code(q["code"], language="python")
            opts = list(q["options"])
            random.Random(q["q"] + (q["code"] or "")).shuffle(opts)
            answers.append(st.radio("Your answer", opts, index=None, key=f"q{lid}_{attempt_no}_{i}",
                                    label_visibility="collapsed"))
        submitted = st.form_submit_button("Submit quiz")

    if submitted:
        if any(x is None for x in answers):
            st.warning("Please answer every question before submitting.")
        else:
            correct = [x == q["answer"] for x, q in zip(answers, lesson["questions"])]
            score = 100 * sum(correct) / len(correct)
            minutes = min(max((time.time() - timers.pop((topic, diff), time.time())) / 60, 0.5), 60)
            n = db.save_quiz_result(user["user_id"], topic, diff, score, minutes)
            st.session_state["last_quiz"] = {
                "topic": topic, "diff": diff, "score": score, "attempt": n, "minutes": minutes,
                "review": [(q, x, ok) for q, x, ok in zip(lesson["questions"], answers, correct)]}

    last = st.session_state.get("last_quiz")
    if last and (last["topic"], last["diff"]) == (topic, diff):
        show_quiz_result(user, last)


def show_quiz_result(user, last):
    st.divider()
    passed = last["score"] >= PASS_MARK
    (st.success if passed else st.warning)(
        f"Score: {last['score']:.0f}%  ·  attempt {last['attempt']}  ·  {last['minutes']:.1f} min  ·  "
        + ("Passed" if passed else f"Below the {PASS_MARK}% pass mark"))
    with st.expander("Review your answers", expanded=not passed):
        for i, (q, given, ok) in enumerate(last["review"], 1):
            st.markdown(f"{'✅' if ok else '❌'} **Q{i}.** {q['q']}")
            if q["code"]:
                st.code(q["code"], language="python")
            st.markdown(f"Your answer: `{given}`" + ("" if ok else f"  ·  Correct: `{q['answer']}`"))
            st.caption(q["why"])
    a = analyse(user["user_id"])  # updated level + recommendation after this quiz
    if a["prediction"]:
        st.markdown(f"**Your level is now:** " + badge(a["prediction"]["level"], charts.LEVEL_COLORS[a["prediction"]["level"]])
                    + f' <span class="muted">({a["prediction"]["confidence"] * 100:.0f}% confidence)</span>',
                    unsafe_allow_html=True)
    recommendation_card(a["recommendation"], "after_quiz")


def page_prediction(user):
    import predictor
    st.header("Performance Prediction")
    st.caption("Enter academic and activity data, and the trained ANN shows the estimated level. "
               "Change the sliders to see what would happen (what-if).")
    a = analyse(user["user_id"])
    defaults = a["features"] or {"avg_quiz_score": 60, "last_quiz_score": 60, "time_spent": 120,
                                 "attempts_per_quiz": 2.0, "lessons_completed": 50,
                                 "login_frequency": 4, "days_since_last_login": 2}
    ranges = {"avg_quiz_score": (0.0, 100.0, 1.0), "last_quiz_score": (0.0, 100.0, 1.0),
              "time_spent": (10.0, 600.0, 5.0), "attempts_per_quiz": (1.0, 5.0, 0.1),
              "lessons_completed": (0.0, 100.0, 1.0), "login_frequency": (0.0, 14.0, 0.5),
              "days_since_last_login": (0.0, 30.0, 1.0)}

    def reset():
        for k in FEATURES:
            st.session_state[f"pp_{k}"] = float(min(max(defaults[k], ranges[k][0]), ranges[k][1]))

    if "pp_init" not in st.session_state or st.session_state.get("pp_user") != user["user_id"]:
        reset()
        st.session_state["pp_init"], st.session_state["pp_user"] = True, user["user_id"]

    cols = st.columns(3)
    values = {}
    for i, k in enumerate(FEATURES):
        lo, hi, step = ranges[k]
        values[k] = cols[i % 3].slider(FEATURE_LABELS[k], lo, hi, step=step, key=f"pp_{k}")
    st.button("Reset to my real data", on_click=reset)

    pred = predictor.predict(values)
    st.divider()
    r1, r2, r3 = st.columns([1, 1.3, 1.3])
    with r1:
        idx = 100 * (0.5 * pred["probs"]["Average"] + pred["probs"]["Strong"])
        st.markdown("**Predicted level**")
        st.markdown(badge(pred["level"], charts.LEVEL_COLORS[pred["level"]]), unsafe_allow_html=True)
        st.metric("Performance index", f"{idx:.0f} / 100",
                  help="0.5 x P(Average) + 1.0 x P(Strong), shown as a 0-100 scale for easy reading.")
        st.metric("Model confidence", f"{pred['confidence'] * 100:.0f}%")
    r2.plotly_chart(charts.probabilities(pred["probs"]))
    r3.plotly_chart(charts.factors(pred["factors"]))
    st.caption("Normalised factors compare each value with the typical trainee in the training data "
               "(attempts and idle days are flipped so that green is always better).")


def page_warning(user):
    st.header("Early Warning & Recommendations")
    a = analyse(user["user_id"])
    pred, feats = a["prediction"], a["features"]
    if not pred:
        st.info("Take at least one quiz so the system can analyse your performance.")
        recommendation_card(a["recommendation"], "warn_start")
        return

    g, s = st.columns([1, 1.4])
    g.plotly_chart(charts.risk_gauge(pred["probs"]["Weak"], pred["risk"]))
    with s:
        st.subheader("What the system noticed")
        for kind, title, text in pred["suggestions"]:
            st.markdown(f'<div class="card {kind}"><h4>{title}</h4><div class="muted">{text}</div></div>',
                        unsafe_allow_html=True)

    st.subheader("Your adaptive learning path")
    recommendation_card(a["recommendation"], "warn_next")
    st.markdown("**Also recommended for a " + pred["level"] + " trainee:**")
    for item in a["recommendation"]["extras"]:
        st.markdown(f"- {item}")


# --------------------------------------------------------------------------- #
# Admin pages
# --------------------------------------------------------------------------- #
def refresh_all():
    for uid in db.list_trainees()["user_id"]:
        analyse(int(uid))


def page_admin_overview():
    st.header("Admin Overview")
    refresh_all()
    df = db.list_trainees()
    if df.empty:
        st.info("No trainees yet.")
        return
    counts = df["predicted_level"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trainees", len(df))
    for c, lvl in zip((c2, c3, c4), LEVELS):
        c.metric(lvl, counts.get(lvl, 0))
    left, right = st.columns([1, 1.5])
    left.plotly_chart(charts.level_counts(counts))
    with right:
        st.subheader("All trainees")
        show = df[["name", "email", "predicted_level", "confidence", "lessons_completed", "last_updated"]].copy()
        show["confidence"] = (show["confidence"] * 100).round(0)
        show.columns = ["Name", "Email", "Predicted level", "Confidence (%)", "Topics completed (%)", "Updated"]
        st.dataframe(show, hide_index=True)
    st.caption("Trainees with no quiz result have no prediction yet.")


def page_admin_detail():
    st.header("Trainee Detail")
    df = db.list_trainees()
    if df.empty:
        st.info("No trainees yet.")
        return
    choice = st.selectbox("Trainee", df["user_id"], format_func=lambda i: df.set_index("user_id").loc[i, "name"])
    a = analyse(int(choice))
    if not a["prediction"]:
        st.info("This trainee has not taken a quiz yet.")
        return
    p, f = a["prediction"], a["features"]
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Predicted level**<br>" + badge(p["level"], charts.LEVEL_COLORS[p["level"]]), unsafe_allow_html=True)
    c2.markdown("**Risk**<br>" + badge(p["risk"], charts.RISK_COLORS[p["risk"]]), unsafe_allow_html=True)
    c3.metric("Next lesson", f"{a['recommendation']['topic']} ({a['recommendation']['difficulty']})")
    l, r = st.columns(2)
    l.plotly_chart(charts.topic_scores(a["results"]))
    r.plotly_chart(charts.score_progress(a["results"]))
    st.subheader("ANN input features")
    st.dataframe(pd.DataFrame({"Feature": [FEATURE_LABELS[k] for k in FEATURES], "Value": [f[k] for k in FEATURES]}),
                 hide_index=True)
    st.subheader("Quiz history")
    st.dataframe(a["results"].drop(columns="result_id").iloc[::-1], hide_index=True)


def page_admin_model():
    import json
    st.header("Model Evaluation")
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backend", "TensorFlow/Keras" if info["backend"] == "keras" else "scikit-learn MLP")
    c2.metric("Epochs run", info["epochs_run"])
    c3.metric("Training records", info["n_train"])
    c4.metric("Test accuracy", f"{info['test_accuracy'] * 100:.1f}%")
    table = REPORT_DIR / "model_comparison.csv"
    if table.exists():
        st.subheader("Model comparison (test set)")
        st.dataframe(pd.read_csv(table), hide_index=True)
    curves = REPORT_DIR / "training_curves.png"
    if curves.exists():
        st.subheader("Training and validation curves")
        st.image(str(curves), width=760)
    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, name in zip(cols, ["logistic", "decision", "ann"]):
        p = REPORT_DIR / f"confusion_matrix_{name}.png"
        if p.exists():
            col.image(str(p))
    st.divider()
    if st.button("Retrain the model on a fresh dataset"):
        from config import DATASET_PATH
        DATASET_PATH.unlink(missing_ok=True)
        train_model_ui()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if "user" not in st.session_state:
        login_page()
        return
    if not model_ready():
        st.title("Model not trained yet")
        st.write("Train the ANN once (about 1 minute). You can also run `python train_all.py` in the terminal.")
        if st.button("Train model now"):
            train_model_ui()
        return

    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"### 🎓 {user['name']}")
        st.caption(user["role"].capitalize())
        if user["role"] == "admin":
            page = st.radio("Menu", ["Overview", "Trainee Detail", "Model Evaluation"], key="nav_admin")
        else:
            page = st.radio("Menu", ["Dashboard", "Learn & Quiz", "Performance Prediction",
                                     "Early Warning"], key="nav_trainee")
        st.divider()
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    {"Dashboard": lambda: page_dashboard(user), "Learn & Quiz": lambda: page_learn(user),
     "Performance Prediction": lambda: page_prediction(user), "Early Warning": lambda: page_warning(user),
     "Overview": page_admin_overview, "Trainee Detail": page_admin_detail,
     "Model Evaluation": page_admin_model}[page]()


main()
