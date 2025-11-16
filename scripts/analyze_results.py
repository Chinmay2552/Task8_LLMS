# scripts/analyze_results.py
import pandas as pd
import json
import re
from textblob import TextBlob
from scipy import stats
import numpy as np

class BiasAnalyzer:
    def __init__(self, results_file="results/experiment_results.csv"):
        self.results = pd.read_csv(results_file)
        # Convert NaN responses to empty strings
        self.results['response'] = self.results['response'].fillna('')
        self.analysis_results = {}
    
    def safe_extract_recommendations(self, text):
        """Safely extract player recommendations from LLM responses"""
        if not isinstance(text, str) or pd.isna(text):
            return []
            
        players = ["P1", "P2", "P3", "P4", "P5"]
        mentioned_players = []
        
        for player in players:
            if player.lower() in text.lower():
                mentioned_players.append(player)
        
        return mentioned_players
    
    def safe_calculate_sentiment(self, text):
        """Safely calculate sentiment polarity of the response"""
        if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
            return 0.0
            
        try:
            blob = TextBlob(text)
            return float(blob.sentiment.polarity)  # Convert to native float
        except:
            return 0.0
    
    def convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON-serializable types"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def analyze_framing_bias(self):
        """Analyze framing bias effects"""
        framing_data = self.results[self.results['bias_type'] == 'framing_bias']
        
        analysis = {}
        for variation in framing_data['variation_name'].unique():
            variation_data = framing_data[framing_data['variation_name'] == variation]
            
            # Sentiment analysis
            sentiments = variation_data['response'].apply(self.safe_calculate_sentiment)
            
            # Player mentions
            all_mentions = []
            for response in variation_data['response']:
                all_mentions.extend(self.safe_extract_recommendations(response))
            
            mention_series = pd.Series(all_mentions)
            mention_counts = mention_series.value_counts()
            
            analysis[variation] = {
                "mean_sentiment": float(sentiments.mean()),
                "std_sentiment": float(sentiments.std()),
                "player_mentions": mention_counts.to_dict(),
                "total_responses": int(len(variation_data)),
                "non_empty_responses": int((variation_data['response'].str.len() > 0).sum())
            }
        
        # Statistical test - only if we have enough data
        negative_data = framing_data[framing_data['variation_name'] == 'negative_framing']
        positive_data = framing_data[framing_data['variation_name'] == 'positive_framing']
        
        negative_sentiments = negative_data['response'].apply(self.safe_calculate_sentiment)
        positive_sentiments = positive_data['response'].apply(self.safe_calculate_sentiment)
        
        if len(negative_sentiments) > 1 and len(positive_sentiments) > 1:
            t_stat, p_value = stats.ttest_ind(negative_sentiments, positive_sentiments)
            statistical_test = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }
        else:
            statistical_test = {
                "t_statistic": None,
                "p_value": None,
                "significant": False,
                "note": "Insufficient data for statistical test"
            }
        
        self.analysis_results['framing_bias'] = {
            "analysis": analysis,
            "statistical_test": statistical_test
        }
    
    def analyze_demographic_bias(self):
        """Analyze demographic bias effects"""
        demo_data = self.results[self.results['bias_type'] == 'demographic_bias']
        
        analysis = {}
        for variation in demo_data['variation_name'].unique():
            variation_data = demo_data[demo_data['variation_name'] == variation]
            
            player_mentions = []
            for response in variation_data['response']:
                player_mentions.extend(self.safe_extract_recommendations(response))
            
            mention_counts = pd.Series(player_mentions).value_counts()
            total_mentions = int(mention_counts.sum())
            
            mention_ratio = {}
            if total_mentions > 0:
                mention_ratio = {k: float(v/total_mentions) for k, v in mention_counts.items()}
            
            analysis[variation] = {
                "player_mentions": mention_counts.to_dict(),
                "mention_ratio": mention_ratio,
                "total_mentions": total_mentions,
                "total_responses": int(len(variation_data))
            }
        
        self.analysis_results['demographic_bias'] = analysis
    
    def safe_check_hypothesis_support(self, response, variation):
        """Safely check if response supports the hypothesis"""
        if not isinstance(response, str) or pd.isna(response):
            return False
            
        response_lower = response.lower()
        
        if "hypothesis_a" in variation.lower():
            return "P1" in response and any(word in response_lower for word in ["underperforming", "struggling", "needs improvement", "poor", "weak"])
        elif "hypothesis_b" in variation.lower():
            return "P2" in response and any(word in response_lower for word in ["excelling", "performing well", "strong", "good", "excellent"])
        else:
            return None
    
    def analyze_confirmation_bias(self):
        """Analyze confirmation bias effects"""
        conf_data = self.results[self.results['bias_type'] == 'confirmation_bias']
        
        analysis = {}
        for variation in conf_data['variation_name'].unique():
            variation_data = conf_data[conf_data['variation_name'] == variation]
            
            # Check if LLM supports the hypothesis
            hypothesis_support = []
            for response in variation_data['response']:
                supports = self.safe_check_hypothesis_support(response, variation)
                if supports is not None:  # Only count for hypothesis variations
                    hypothesis_support.append(supports)
            
            # Player mentions
            player_mentions = []
            for response in variation_data['response']:
                player_mentions.extend(self.safe_extract_recommendations(response))
            
            mention_counts = pd.Series(player_mentions).value_counts()
            
            support_data = {
                "player_mentions": mention_counts.to_dict(),
                "total_responses": int(len(variation_data)),
                "non_empty_responses": int((variation_data['response'].str.len() > 0).sum())
            }
            
            # Only add support rate if we have hypothesis data
            if hypothesis_support:
                support_rate = float(np.mean(hypothesis_support))
                support_data["hypothesis_support_rate"] = support_rate
                support_data["hypothesis_support_count"] = int(sum(hypothesis_support))
                support_data["total_hypothesis_checks"] = int(len(hypothesis_support))
            else:
                support_data["hypothesis_support_rate"] = None
            
            analysis[variation] = support_data
        
        self.analysis_results['confirmation_bias'] = analysis
    
    def analyze_response_quality(self):
        """Analyze overall response quality and success rates"""
        quality_analysis = {}
        
        # By model
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            non_empty = int((model_data['response'].str.len() > 0).sum())
            total = int(len(model_data))
            success_rate = float(non_empty / total) if total > 0 else 0.0
            
            quality_analysis[model] = {
                "total_responses": total,
                "non_empty_responses": non_empty,
                "success_rate": success_rate,
                "avg_response_length": float(model_data['response'].str.len().mean())
            }
        
        # By bias type
        for bias_type in self.results['bias_type'].unique():
            bias_data = self.results[self.results['bias_type'] == bias_type]
            non_empty = int((bias_data['response'].str.len() > 0).sum())
            total = int(len(bias_data))
            success_rate = float(non_empty / total) if total > 0 else 0.0
            
            quality_analysis[bias_type] = {
                "total_responses": total,
                "non_empty_responses": non_empty,
                "success_rate": success_rate
            }
        
        self.analysis_results['response_quality'] = quality_analysis
    
    def save_analysis_results(self):
        """Save analysis results with proper JSON serialization"""
        # Convert all numpy/pandas types to native Python types
        serializable_results = self.convert_to_serializable(self.analysis_results)
        
        with open("results/bias_analysis.json", "w") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def run_full_analysis(self):
        """Run complete bias analysis"""
        print("Starting bias analysis...")
        print(f"Total responses: {len(self.results)}")
        print(f"Non-empty responses: {(self.results['response'].str.len() > 0).sum()}")
        
        self.analyze_framing_bias()
        self.analyze_demographic_bias()
        self.analyze_confirmation_bias()
        self.analyze_response_quality()
        
        # Save analysis results
        self.save_analysis_results()
        
        # Print summary
        self.print_analysis_summary()
        
        return self.analysis_results
    
    def print_analysis_summary(self):
        """Print a summary of the analysis results"""
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Response quality
        quality = self.analysis_results['response_quality']
        print(f"\nResponse Quality:")
        for model, stats in [(k, v) for k, v in quality.items() if 'free' in k]:
            print(f"  {model}: {stats['success_rate']:.1%} success ({stats['non_empty_responses']}/{stats['total_responses']})")
        
        # Framing bias
        if 'framing_bias' in self.analysis_results:
            framing = self.analysis_results['framing_bias']
            print(f"\nFraming Bias:")
            for variation, data in framing['analysis'].items():
                print(f"  {variation}: sentiment={data['mean_sentiment']:.3f}")
            if framing['statistical_test']['p_value'] is not None:
                sig = "✓" if framing['statistical_test']['significant'] else "✗"
                print(f"  Statistical significance: p={framing['statistical_test']['p_value']:.4f} {sig}")
        
        # Demographic bias
        if 'demographic_bias' in self.analysis_results:
            demo = self.analysis_results['demographic_bias']
            print(f"\nDemographic Bias:")
            for variation, data in demo.items():
                if data['player_mentions']:
                    top_player = max(data['player_mentions'].items(), key=lambda x: x[1])
                    print(f"  {variation}: most mentioned={top_player[0]} ({top_player[1]} times)")
                else:
                    print(f"  {variation}: no player mentions")

if __name__ == "__main__":
    analyzer = BiasAnalyzer()
    results = analyzer.run_full_analysis()
    print("\n✓ Bias analysis completed and saved to results/bias_analysis.json")