"""Adaptive engine: turns the ANN prediction into the next learning step
(Section 5.6 / Table 4 of the report) and builds early-warning suggestions."""
import pandas as pd

from config import DIFFICULTIES, LEVEL_TO_DIFFICULTY, PASS_MARK, REVISE_MARK
from lesson_content import TOPICS

EXTRAS = {
    "Weak": ["Revise the previous topics", "Take an extra practice quiz", "Move on only after scoring 60% or more"],
    "Average": ["Continue with normal progression", "Retake any quiz below 60%"],
    "Strong": ["Try the optional challenge quiz", "Study advanced topics at Hard level"],
}


def _easier(difficulty: str) -> str:
    return DIFFICULTIES[max(DIFFICULTIES.index(difficulty) - 1, 0)]


def _topic_done(results: pd.DataFrame, topic: str, difficulty: str) -> bool:
    """A topic is done when it was passed at the target difficulty or a harder one."""
    ok = results[(results["topic"] == topic) & (results["score"] >= PASS_MARK)]
    return any(DIFFICULTIES.index(d) >= DIFFICULTIES.index(difficulty) for d in ok["difficulty"])


def recommend(level, results: pd.DataFrame) -> dict:
    """Return the next lesson: {action, topic, difficulty, message, extras}."""
    if level is None or results.empty:
        return {"action": "Start", "topic": TOPICS[0], "difficulty": "Medium",
                "message": "Welcome! Start with a placement lesson. The system needs at least one quiz "
                           "result before it can predict your level.",
                "extras": ["Finish the quiz to unlock your personal level prediction"]}

    target = LEVEL_TO_DIFFICULTY[level]
    last = results.iloc[-1]
    latest_by_topic = results.groupby("topic").tail(1).set_index("topic")["score"]

    # 1) A topic where the latest score is very low -> revise that topic only
    low = latest_by_topic[latest_by_topic < REVISE_MARK]
    if not low.empty:
        topic = low.idxmin()
        diff = target if level == "Strong" else _easier(target)
        return {"action": "Revise", "topic": topic, "difficulty": diff,
                "message": f"Your latest score in '{topic}' was {low.min():.0f}%. Revise this topic at "
                           f"{diff} level before moving on.",
                "extras": EXTRAS[level]}

    # 2) The lesson just taken was below the pass mark -> try again, one level easier
    if last["score"] < PASS_MARK:
        diff = _easier(last["difficulty"])
        return {"action": "Revise", "topic": last["topic"], "difficulty": diff,
                "message": f"You scored {last['score']:.0f}% in '{last['topic']}' ({last['difficulty']}). "
                           f"Try it again at {diff} level to build confidence.",
                "extras": EXTRAS[level]}

    # 3) Next topic that is not yet done at the recommended difficulty
    for topic in TOPICS:
        if not _topic_done(results, topic, target):
            return {"action": "Continue", "topic": topic, "difficulty": target,
                    "message": f"Your predicted level is {level}. Next: '{topic}' at {target} level.",
                    "extras": EXTRAS[level]}

    # 4) Everything done at this level
    if level == "Strong":
        weakest = results.groupby("topic")["score"].mean().idxmin()
        return {"action": "Challenge", "topic": weakest, "difficulty": "Hard",
                "message": "You completed every topic at Hard level. Try the challenge quiz on your "
                           f"lowest-scoring topic, '{weakest}'.", "extras": EXTRAS[level]}
    harder = DIFFICULTIES[min(DIFFICULTIES.index(target) + 1, 2)]
    return {"action": "Challenge", "topic": TOPICS[0], "difficulty": harder,
            "message": f"You completed all topics at {target} level. Great work! Try {harder} level "
                       "to keep improving.", "extras": EXTRAS[level]}


def risk_level(prob_weak: float) -> str:
    if prob_weak >= 0.50:
        return "High"
    if prob_weak >= 0.25:
        return "Medium"
    return "Low"


def build_suggestions(f: dict, level: str, risk: str) -> list:
    """Rule-based early-warning messages from the trainee's features.
    Each item: (kind, title, text) where kind is 'warning', 'good' or 'tip'."""
    s = []
    if risk == "High":
        s.append(("warning", "High risk of falling behind",
                  "Your pattern matches trainees who struggle. Revise the Easy lessons and ask your instructor for help."))
    elif risk == "Medium":
        s.append(("warning", "Some risk of difficulty", "A few indicators are weak. Small changes now can prevent problems later."))
    if f["avg_quiz_score"] < 50:
        s.append(("warning", "Low average score", "Re-read the lesson notes before each quiz and review the explanations of wrong answers."))
    elif f["avg_quiz_score"] >= 80:
        s.append(("good", "Strong quiz scores", "Your average score is above 80%. Keep it up and try the harder lessons."))
    if f["last_quiz_score"] < f["avg_quiz_score"] - 15:
        s.append(("warning", "Recent drop in performance", "Your last quiz was much lower than your average. Revisit that topic."))
    if f["attempts_per_quiz"] >= 3:
        s.append(("tip", "Many attempts needed", "Study the lesson fully before starting a quiz instead of guessing."))
    if f["days_since_last_login"] >= 7:
        s.append(("warning", "Inactive for a while", f"You have not logged in for {f['days_since_last_login']} days. Short daily sessions work better than long gaps."))
    if f["login_frequency"] < 2:
        s.append(("tip", "Low login frequency", "Aim for at least 3 study sessions per week."))
    elif f["login_frequency"] >= 4:
        s.append(("good", "Regular study habit", "You log in regularly, which strongly helps retention."))
    if f["lessons_completed"] < 30:
        s.append(("tip", "Few topics completed", "Plan your assignments: finish one topic per study session."))
    elif f["lessons_completed"] >= 75:
        s.append(("good", "Good course progress", "Most topics are completed."))
    if not s:
        s.append(("good", "On track", f"Your indicators look healthy for a {level} trainee."))
    return s
