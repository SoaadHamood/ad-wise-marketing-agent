import json
from app.llm import ask_llm

def run_smart_diagnosis(user_text: str, current_signals_path: str = "outputs/relational_ads_signals.json", historical_signals_path: str = "outputs/social_media_advertising_signals.json") -> dict:
    """
    Worker 1: Strictly analyzes metrics and current performance.
    """
    try:
        with open(current_signals_path, "r") as f:
            current_data = json.load(f)
        with open(historical_signals_path, "r") as f:
            historical_data = json.load(f)
    except FileNotFoundError as e:
        return {"error": f"Missing signal file: {e}."}

    curr_kpis = current_data.get("overall_kpis", {})
    curr_insights = [i.get("message", "") for i in current_data.get("rule_based_insights", []) if isinstance(i, dict)]
    hist_kpis = historical_data.get("overall_kpis", {})

    system_prompt = (
        "You are the Lead Data Analyst for Ad-Wise. Your role is to analyze current campaign performance.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Diagnose the click-engagement gap using the provided numbers.\n"
        "2. Use historical context as an invisible baseline. DO NOT mention it explicitly.\n"
        "3. ONLY provide a diagnosis and an explanation of the signals. DO NOT suggest strategies or headlines.\n"
        "4. Be objective, professional, and concise."
    )

    user_prompt = (
        f"Context: '{user_text}'\n\n"
        f"--- SIGNALS ---\nKPIs: {json.dumps(curr_kpis)}\nInsights: {json.dumps(curr_insights)}\n"
        f"--- BASELINE ---\nKPIs: {json.dumps(hist_kpis)}"
    )

    analysis_report = ask_llm(system_prompt, user_prompt)

    return {
        "report": analysis_report,
        "kpis_used": curr_kpis,
    }