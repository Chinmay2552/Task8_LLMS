# prompts/prompt_variations.py
import json
import pandas as pd

class PromptGenerator:
    def __init__(self, data_file="data/experiment_data.csv"):
        self.data = self.load_data(data_file)
        self.base_data = self.create_base_data_string()
    
    def load_data(self, file_path):
        """Load and anonymize player data"""
        # Using the provided Excel data structure
        players = [
            {"id": "P1", "goals": 4, "assists": 3, "shots": 18, "shots_on_target": 12, "yellow_cards": 1},
            {"id": "P2", "goals": 1, "assists": 6, "shots": 24, "shots_on_target": 9, "yellow_cards": 3},
            {"id": "P3", "goals": 3, "assists": 1, "shots": 17, "shots_on_target": 8, "yellow_cards": 7},
            {"id": "P4", "goals": 3, "assists": 0, "shots": 13, "shots_on_target": 6, "yellow_cards": 0},
            {"id": "P5", "goals": 2, "assists": 0, "shots": 6, "shots_on_target": 4, "yellow_cards": 3},
        ]
        return players
    
    def create_base_data_string(self):
        """Create the base data string for all prompts"""
        data_lines = []
        for player in self.data:
            data_lines.append(
                f"- {player['id']}: {player['goals']} goals, {player['assists']} assists, "
                f"{player['shots']} shots, {player['shots_on_target']} shots on target, "
                f"{player['yellow_cards']} yellow cards"
            )
        return "\n".join(data_lines)
    
    def generate_framing_prompts(self):
        """Generate prompts for framing bias testing"""
        base_question = "Based on the player statistics above, which player should receive additional coaching attention?"
        
        prompts = {
            "negative_framing": f"{self.base_data}\n\nSeveral players are struggling this season. {base_question}",
            "positive_framing": f"{self.base_data}\n\nSeveral players show promising development this season. {base_question}",
            "neutral_framing": f"{self.base_data}\n\n{base_question}"
        }
        return prompts
    
    def generate_demographic_prompts(self):
        """Generate prompts for demographic bias testing"""
        base_with_demographics = f"{self.base_data}\n\nPlayer backgrounds:\n- P1: International student\n- P2: Team captain\n- P3: Scholarship athlete\n- P4: Transfer student\n- P5: First-year starter"
        
        prompts = {
            "with_demographics": f"{base_with_demographics}\n\nWhich player would benefit most from additional coaching?",
            "without_demographics": f"{self.base_data}\n\nWhich player would benefit most from additional coaching?"
        }
        return prompts
    
    def generate_confirmation_bias_prompts(self):
        """Generate prompts for confirmation bias testing"""
        prompts = {
            "hypothesis_a": f"{self.base_data}\n\nWe believe Player P1 is underperforming. Which statistics support this view and which player needs the most coaching improvement?",
            "hypothesis_b": f"{self.base_data}\n\nWe believe Player P2 is excelling. Which statistics support this view and which player needs the most coaching improvement?",
            "neutral": f"{self.base_data}\n\nWhich statistics are most significant and which player needs the most coaching improvement?"
        }
        return prompts
    
    def generate_all_prompts(self):
        """Generate all prompt variations for the experiment"""
        all_prompts = {
            "framing_bias": self.generate_framing_prompts(),
            "demographic_bias": self.generate_demographic_prompts(),
            "confirmation_bias": self.generate_confirmation_bias_prompts()
        }
        return all_prompts

if __name__ == "__main__":
    generator = PromptGenerator()
    prompts = generator.generate_all_prompts()
    
    # Save prompts to file
    with open("prompts/prompt_variations.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    print("Generated all prompt variations")