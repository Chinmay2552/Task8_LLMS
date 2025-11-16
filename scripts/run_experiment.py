# scripts/run_experiment.py
import os
import requests
import json
import time
import random
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

class RateLimitedLLMExperiment:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.models = [
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ]
        
        # Rate limiting settings for free models
        self.requests_per_minute = 20
        self.min_delay = 60 / self.requests_per_minute  # 3 seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.minute_window_start = time.time()
    
    def query_llm(self, prompt, model, seed=42, temperature=0.7):
        """Query an LLM through Open Router"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error querying {model}: {e}")
            return None
    
    def wait_for_rate_limit(self):
        """Implement rate limiting with exponential backoff"""
        current_time = time.time()
        
        # Reset minute counter if we're in a new minute window
        if current_time - self.minute_window_start >= 60:
            self.request_count = 0
            self.minute_window_start = current_time
        
        # Check if we've exceeded the minute limit
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.minute_window_start)
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            self.request_count = 0
            self.minute_window_start = time.time()
        
        # Enforce minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def query_llm_with_retry(self, prompt, model, max_retries=5, seed=42):
        """Query LLM with exponential backoff retry logic - FIXED: added seed parameter"""
        initial_delay = 1
        max_delay = 60
        
        for attempt in range(max_retries):
            try:
                self.wait_for_rate_limit()
                response = self.query_llm(prompt, model, seed=seed)
                
                if response is not None:
                    return response
                else:
                    # If we get null response, wait and retry
                    delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Null response from {model}. Retry {attempt + 1}/{max_retries} in {delay:.2f}s")
                    time.sleep(delay)
                    
            except Exception as e:
                delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Error from {model}: {e}. Retry {attempt + 1}/{max_retries} in {delay:.2f}s")
                time.sleep(delay)
        
        print(f"Failed to get response from {model} after {max_retries} attempts")
        return None
    
    def run_experiment(self, prompts_file="prompts/prompt_variations.json"):
        """Run experiment with rate limiting"""
        # Load prompts
        with open(prompts_file, "r") as f:
            all_prompts = json.load(f)
        
        results = []
        total_expected_requests = sum(len(variations) for variations in all_prompts.values()) * len(self.models) * 3
        
        print(f"Total expected requests: {total_expected_requests}")
        print("This will take approximately {:.1f} minutes to complete".format(total_expected_requests * self.min_delay / 60))
        
        for bias_type, prompt_variations in all_prompts.items():
            for variation_name, prompt in prompt_variations.items():
                for model in self.models:
                    for run in range(3):  # 3 runs per prompt
                        print(f"Running: {bias_type} - {variation_name} - {model} - Run {run+1}")
                        
                        # FIX: Pass seed as keyword argument to query_llm_with_retry
                        response = self.query_llm_with_retry(prompt, model, seed=run)
                        
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "bias_type": bias_type,
                            "variation_name": variation_name,
                            "model": model,
                            "prompt": prompt,
                            "response": response,
                            "run_id": run,
                            "seed": run
                        }
                        results.append(result)
                        
                        # Save incremental results in case of failure
                        if len(results) % 5 == 0:
                            df = pd.DataFrame(results)
                            df.to_csv("results/experiment_results_partial.csv", index=False)
        
        # Save final results
        df = pd.DataFrame(results)
        df.to_csv("results/experiment_results.csv", index=False)
        
        # Count successful vs failed responses
        successful = len([r for r in results if r['response'] is not None])
        print(f"Experiment completed: {successful}/{len(results)} successful responses")
        
        return results

if __name__ == "__main__":
    experiment = RateLimitedLLMExperiment()
    results = experiment.run_experiment()
    print(f"Experiment completed with {len(results)} responses")