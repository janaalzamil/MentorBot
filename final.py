import nest_asyncio
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

nest_asyncio.apply()

# Configure model
model = OpenAIModel(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.together.xyz/v1",
    api_key=open('secret').read(),
)


def generate_related_terms(skill: str) -> set:
    """Generate related skills/synonyms using AI"""
    try:
        agent = Agent(
            model,
            system_prompt="""Generate 5-7 technical terms closely related to the provided skill. 
                          Return ONLY comma-separated lowercase keywords. No explanations."""
        )
        response = agent.run_sync(skill).data.lower()
        terms = {skill.strip() for skill in response.split(",")}
        terms.add(skill.strip().lower())  # Include original skill
        return terms
    except Exception as e:
        st.error(f"Error generating terms for {skill}: {str(e)}")
        return {skill.strip().lower()}


def normalize_skills(skills_str: str) -> list:
    """Clean and normalize skills from model response"""
    cleaned = ''.join([c if c.isalpha() or c == ',' else ' ' for c in skills_str.lower()])
    skills = [skill.strip().replace(" ", "") for skill in cleaned.split(",") if skill.strip()]
    return skills[:10]


def calculate_match(user_skills: dict, job_skills: list) -> tuple:
    """Calculate skill match percentage with semantic matching"""
    matches = []
    for job_skill in job_skills:
        for uskill, variations in user_skills.items():
            if any(var in job_skill for var in variations):
                matches.append(job_skill)
                break  # Avoid duplicate matches
    total_skills = max(len(job_skills), 1)
    return (len(matches) / total_skills) * 100, matches


# Streamlit UI
st.set_page_config(page_title="MentorBot", layout="centered")
st.title("Welcome to Mentor Bot!")
st.markdown("""
    This app helps you find the careers that match your skills. 
    Simply input the careers you're considering and the skills you have. 
    The app will compare your skills with the key skills required for each career and show the match percentage.
""")

# Input sections
col1, col2 = st.columns(2)
with col1:
    careers_input = st.text_input("Careers (comma-separated):")
with col2:
    skills_input = st.text_input("Your Skills (comma-separated):")

if st.button("Analyze Matches") and careers_input and skills_input:
    # Preprocess user skills with generated variations
    base_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
    user_skills = {skill: generate_related_terms(skill) for skill in base_skills}

    results = []

    for career in [c.strip() for c in careers_input.split(",") if c.strip()]:
        try:
            # Get required skills
            agent = Agent(
                model,
                system_prompt="""List 10 CORE TECHNICAL SKILLS as comma-separated lowercase keywords 
                              for this job. Only technical terms, no explanations."""
            )
            response = agent.run_sync(f"Skills needed for {career}:").data
            job_skills = normalize_skills(response)

            # Calculate matches
            match_percent, matched_skills = calculate_match(user_skills, job_skills)

            results.append({
                "career": career,
                "match": match_percent,
                "matched": matched_skills,
                "required": job_skills
            })

        except Exception as e:
            st.error(f"Error processing {career}: {str(e)}")

    # Display results
    if results:
        st.subheader("Analysis Results")
        for result in sorted(results, key=lambda x: x['match'], reverse=True):
            with st.expander(f"{result['career']} - {result['match']:.1f}% Match"):
                st.metric("Match Percentage", f"{result['match']:.1f}%")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Matching Skills**")
                    st.write("\n".join(result['matched']) or "No direct matches")

                with col_b:
                    st.write("**Required Skills**")
                    st.write(", ".join(result['required']))
    else:
        st.warning("No valid careers to analyze")

else:
    st.info("Please enter at least one career and your skills to begin analysis")