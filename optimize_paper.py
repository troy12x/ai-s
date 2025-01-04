"""
Script to generate high-scoring research papers using the paper optimizer.
"""

import os
import json
from datetime import datetime
from aider.coders import Coder
from aider.models import Model

from ai_scientist.paper_optimizer import optimize_paper
from ai_scientist.generate_ideas import generate_ideas
from ai_scientist.llm import create_client, AVAILABLE_LLMS

def main():
    # Setup
    model = "gpt-4o"  # Use most capable model for best results
    client = create_client(model)
    coder = Coder.create(model=Model(model), client=client)
    
    # Generate initial research idea
    ideas = generate_ideas(
        model=model,
        client=client,
        temperature=0.7
    )
    idea = ideas[0]  # Take first idea
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"optimized_paper_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Optimize paper
    final_review, history = optimize_paper(
        idea=idea,
        folder_name=folder_name,
        coder=coder,
        model=model,
        client=client,
        target_score=3.5,  # Target high score (3.5/4.0)
        max_iterations=3
    )
    
    # Save optimization history
    history_path = os.path.join(folder_name, "optimization_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nPaper optimization complete!")
    print(f"Final average score: {sum(final_review[k] for k in ['Originality', 'Quality', 'Clarity', 'Significance'])/4:.2f}")
    print(f"Results saved in: {folder_name}")

if __name__ == "__main__":
    main()
