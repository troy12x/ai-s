"""
Paper Optimizer Module

This module implements an iterative optimization loop to generate high-scoring research papers.
It uses the review system as feedback to improve paper quality through multiple iterations.
"""

import os
from typing import Dict, List, Optional, Tuple

from ai_scientist.perform_review import (
    perform_review,
    get_meta_review,
    reviewer_system_prompt_neg,
)
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers

IMPROVEMENT_PROMPT = """You are an expert ML researcher. Given a paper review, suggest specific improvements to address the weaknesses and enhance the strengths.
Focus on making changes that will improve the paper's scores in:
- Originality (novelty and innovation)
- Quality (technical soundness)
- Clarity (writing and presentation)
- Significance (impact and importance)

Current Review:
{review}

Respond in JSON format with the following fields:
{
    "improvements": [
        {
            "aspect": "string (one of: methodology, experiments, writing, theory)",
            "suggestion": "specific suggestion",
            "expected_impact": "how this will improve the paper"
        }
    ],
    "priority": "ordered list of improvements from highest to lowest impact"
}
"""

def calculate_paper_score(review: Dict) -> float:
    """Calculate overall paper score from individual metrics."""
    metrics = ['Originality', 'Quality', 'Clarity', 'Significance']
    return sum(review.get(metric, 1) for metric in metrics) / len(metrics)

def get_improvement_suggestions(
    review: Dict,
    model: str,
    client: any,
    temperature: float = 0.7
) -> Dict:
    """Generate specific suggestions to improve the paper based on review."""
    prompt = IMPROVEMENT_PROMPT.format(review=review)
    response = get_response_from_llm(
        model=model,
        client=client,
        prompt=prompt,
        temperature=temperature
    )
    return extract_json_between_markers(response)

def optimize_paper(
    idea: Dict,
    folder_name: str,
    coder: any,
    model: str,
    client: any,
    target_score: float = 3.5,
    max_iterations: int = 3,
    temperature: float = 0.7
) -> Tuple[Dict, List[Dict]]:
    """
    Iteratively improve a research paper until it reaches the target score.
    
    Args:
        idea: Initial research idea
        folder_name: Output directory
        coder: Aider coder instance
        model: LLM model to use
        client: LLM client
        target_score: Target average score (1-4 scale)
        max_iterations: Maximum optimization iterations
        temperature: Temperature for LLM sampling
    
    Returns:
        Tuple of (final_review, improvement_history)
    """
    improvement_history = []
    current_score = 0
    iteration = 0
    
    while current_score < target_score and iteration < max_iterations:
        # Generate/update paper
        perform_writeup(
            idea=idea,
            folder_name=folder_name,
            coder=coder,
            cite_client=client,
            cite_model=model
        )
        
        # Get ensemble review
        reviews = []
        for _ in range(3):  # Use 3 reviewers for more reliable feedback
            review = perform_review(
                text=folder_name,
                model=model,
                client=client,
                num_reflections=2,  # More reflections for better review quality
                temperature=temperature,
                reviewer_system_prompt=reviewer_system_prompt_neg  # Use negative bias for stricter review
            )
            reviews.append(review)
        
        # Get meta-review
        meta_review = get_meta_review(
            model=model,
            client=client,
            temperature=temperature,
            reviews=reviews
        )
        
        # Calculate current score
        current_score = calculate_paper_score(meta_review)
        
        # Get improvement suggestions
        if current_score < target_score:
            improvements = get_improvement_suggestions(
                review=meta_review,
                model=model,
                client=client,
                temperature=temperature
            )
            
            # Update idea with improvements
            idea['improvements'] = improvements['improvements']
            idea['priority'] = improvements['priority']
        
        # Record history
        improvement_history.append({
            'iteration': iteration,
            'score': current_score,
            'review': meta_review,
            'improvements': improvements if current_score < target_score else None
        })
        
        iteration += 1
    
    return meta_review, improvement_history
