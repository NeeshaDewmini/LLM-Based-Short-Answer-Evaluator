import os
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY not found. Add it to .env locally or Streamlit secrets when deploying.")
    st.stop()

client = Groq(api_key=api_key)

st.set_page_config(page_title="LLM Answer Evaluator", page_icon="🧠", layout="centered")

st.title("🧠 LLM-Based Short Answer Evaluator")
st.write(
    "A beginner-friendly AI assessment tool that uses an LLM to evaluate short answers, "
    "provide a score, explain the result, and suggest improvements."
)

question = st.text_area("Enter the Question", height=100)
expected_answer = st.text_area("Enter the Expected Answer / Marking Guide", height=150)
student_answer = st.text_area("Enter the Student's Answer", height=150)


def evaluate_answer(question: str, expected_answer: str, student_answer: str) -> dict:
    prompt = f"""
You are an assessment assistant.

Your task is to evaluate a student's answer.

Question:
{question}

Expected Answer / Marking Guide:
{expected_answer}

Student Answer:
{student_answer}

Instructions:
1. Score the answer out of 10
2. Also assign a grade: Excellent, Good, Fair, or Poor
3. Explain the reason for the score
4. Identify strengths
5. Identify missing or incorrect points. 
6. Suggest how the student can improve

Return the response ONLY as valid JSON in this format:
{{
  "score": 0,
  "grade": "",
  "reason": "",
  "strengths": [],
  "missing_points": [],
  "improvement_suggestions": []
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful and fair assessment evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "score": "N/A",
            "grade": "N/A",
            "reason": "Model did not return clean JSON.",
            "strengths": [],
            "missing_points": [],
            "improvement_suggestions": [content]
        }


if st.button("Evaluate Answer"):
    if not question.strip() or not expected_answer.strip() or not student_answer.strip():
        st.warning("Please fill in all fields before evaluating.")
    else:
        with st.spinner("Evaluating answer..."):
            result = evaluate_answer(question, expected_answer, student_answer)

        st.subheader("Evaluation Result")
        st.metric("Score", f"{result.get('score')}/10")
        st.metric("Grade", result.get("grade"))
        st.write("**Reason:**", result.get("reason"))

        st.write("### Strengths")
        strengths = result.get("strengths", [])
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.write("No strengths returned.")

        st.write("### Missing / Incorrect Points")
        missing_points = result.get("missing_points", [])
        if missing_points:
            for item in missing_points:
                st.write(f"- {item}")
        else:
            st.write("No missing points returned.")

        st.write("### Suggestions for Improvement")
        suggestions = result.get("improvement_suggestions", [])
        if suggestions:
            for item in suggestions:
                st.write(f"- {item}")
        else:
            st.write("No suggestions returned.")

        st.download_button(
            label="Download Evaluation Result as JSON",
            data=json.dumps(result, indent=2),
            file_name="evaluation_result.json",
            mime="application/json"
        )