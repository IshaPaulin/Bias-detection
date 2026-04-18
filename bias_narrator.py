import json
import pandas as pd
import google.generativeai as genai

class BiasNarrator:
    """
    Agentic Auditor module utilizing Gemini 2.5 Flash for both Inspector and Explainer roles.
    """
    
    def __init__(self, gemini_api_key: str):
        self.gemini_key = gemini_api_key
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_proxies_with_gemini(self, proxies: dict, metrics: list) -> str:
        """
        Agent A (The Inspector): Uses Gemini to explain socioeconomic meaning of proxies in very simple terms.
        """
        if not self.gemini_key:
            return "Gemini API key not provided."
            
        prompt = f"""
        You are an AI Fairness Inspector speaking to a general audience (like a jury or a non-technical manager). 
        Your job is to explain why the AI using these variables is basically discriminating against people.

        Suspicious proxies detected:
        {json.dumps(proxies, indent=2)}

        Mathematical metrics (for your reference only, do not overly use these numbers):
        {json.dumps(metrics, indent=2)}

        Write a VERY SHORT, simple explanation (max 3-4 sentences). 
        Explain the real-world reason why this variable (like time of day or zip code) is actually just a hidden way to judge someone's class, race, or gender.
        DO NOT use technical jargon, do not mention "Demographic Parity" or "Disparate Impact Ratio". Keep it extremely simple, direct, and short.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"

    def generate_victim_narrative_with_gemini(self, flagged_cases: pd.DataFrame, metrics: list) -> str:
        """
        Agent B (The Explainer): Uses Gemini to generate a short, punchy Victim Narrative.
        Returns a JSON string.
        """
        if not self.gemini_key:
            return json.dumps({"error": "Gemini API key not provided."})
            
        if flagged_cases.empty:
            return json.dumps({"message": "No biased cases found."})
            
        # Sample one compelling case
        sample_case = flagged_cases.iloc[0].to_dict()
        
        prompt = f"""
        You are an investigative journalist writing a very short, punchy story about a victim of AI bias for a general magazine.
        
        Here is the specific applicant who was unfairly rejected because of a hidden bias (even though they were qualified):
        {json.dumps(sample_case, indent=2)}

        Write a short "Victim Narrative". 
        CRITICAL RULES:
        - Keep everything extremely short and simple (max 2 sentences per section).
        - DO NOT use any math, statistics, or acronyms. Talk like a normal human.
        - Explain how silly and unfair it is that changing this one random detail (like their zip code or the hour they applied) would have magically gotten them approved.

        You MUST return your response as a valid JSON object with EXACTLY the following structure, and NO markdown formatting outside the JSON:
        {{
            "victim_profile": "1 short sentence about the victim and their good qualifications.",
            "the_injustice": "1-2 short sentences explaining the crazy fact that they were rejected only because of that specific proxy variable.",
            "systemic_impact": "1 short sentence concluding why this is unfair to everyday people."
        }}
        """
        
        try:
            # Tell Gemini to output strictly JSON
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text
        except Exception as e:
            return json.dumps({"error": f"Error communicating with Gemini API: {str(e)}"})
