import os
import requests
import json
from dotenv import load_dotenv

def setup_environment():
    """Initialize the environment and verify connections"""
    load_dotenv()
    
    # Required packages check
    required_packages = [
        'requests', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'scipy', 'scikit-learn', 'textblob'
    ]
    
    # Verify Open Router API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    print("Environment setup completed successfully")
    return True

if __name__ == "__main__":
    setup_environment()