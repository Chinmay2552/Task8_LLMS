# tests/test_experiment.py
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prompts.prompt_variations import PromptGenerator
from scripts.analyze_results import BiasAnalyzer

def test_prompt_generation():
    """Test that prompts are generated correctly"""
    generator = PromptGenerator()
    prompts = generator.generate_all_prompts()
    
    assert 'framing_bias' in prompts
    assert 'demographic_bias' in prompts
    assert 'confirmation_bias' in prompts
    
    # Test that prompts are different
    framing_prompts = prompts['framing_bias']
    assert framing_prompts['negative_framing'] != framing_prompts['positive_framing']

def test_bias_analyzer():
    """Test bias analysis functionality"""
    analyzer = BiasAnalyzer()
    assert hasattr(analyzer, 'calculate_sentiment')
    assert hasattr(analyzer, 'extract_recommendations')

if __name__ == "__main__":
    pytest.main([__file__])