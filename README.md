# Task 08: Bias Detection in LLM Data Narratives

## Overview
This project investigates systematic biases in LLM-generated data narratives using Syracuse University soccer player statistics.

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Set Open Router API key: `export OPENROUTER_API_KEY=your_key`
3. Run environment setup: `python scripts/setup_environment.py`

## Running the Experiment
1. Generate prompts: `python prompts/prompt_variations.py`
2. Run experiment: `python scripts/run_experiment.py`
3. Analyze results: `python scripts/analyze_results.py`

## Biases Tested
- **Framing Bias**: Negative vs positive framing of player performance
- **Demographic Bias**: Inclusion/exclusion of player background information  
- **Confirmation Bias**: Priming with pre-existing hypotheses

## Results
Findings are documented in `REPORT.md` with statistical tests and visualizations.