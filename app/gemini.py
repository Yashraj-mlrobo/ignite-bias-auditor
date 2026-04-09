import google.generativeai as genai
import pandas as pd
import os

# Set up your API Key
# Recommendation: Use an environment variable for security
genai.configure(api_key="YOUR_FREE_AI_STUDIO_KEY")

def analyze_data_with_gemini(df: pd.DataFrame):
    """
    Sends data summary to Gemini and gets a bias audit response.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # We send a summary (head, types, stats) rather than the whole file 
    # to stay within token limits and maintain privacy.
    data_summary = f"""
    Analyze this dataset for potential bias:
    Columns: {df.columns.tolist()}
    Data Types: {df.dtypes.to_string()}
    Sample Data:
    {df.head(5).to_string()}
    """
    
    try:
        response = model.generate_content(data_summary)
        return {
            "status": "ok",
            "message": "Analysis complete.",
            "result": response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "result": None
        }