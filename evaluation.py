from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import numpy as np
from typing import List, Dict, Any

llm = Ollama(model="phi", temperature=0)

# 🔹 RAGAS-like Metrics (Open Source Implementation)
def context_precision(contexts: List[str], answer: str) -> float:
    """Measures if retrieved contexts are relevant to answer (0-1)"""
    relevant_terms = set(answer.lower().split())
    total_terms = len(relevant_terms)
    
    if total_terms == 0:
        return 1.0
    
    relevant_contexts = 0
    for context in contexts:
        context_terms = set(context.lower().split())
        if relevant_terms.intersection(context_terms):
            relevant_contexts += 1
    
    return min(relevant_contexts / len(contexts), 1.0)

def faithfulness(answer: str, contexts: List[str]) -> float:
    """Checks if answer is grounded in contexts (0-1)"""
    prompt = f"""
    Check if this ANSWER is fully supported by CONTEXTS.
    
    CONTEXTS: {' '.join(contexts)}
    ANSWER: {answer}
    
    Score 1-10: How well is answer supported? (10 = perfect, 0 = hallucinated)
    Just give number:"""
    
    try:
        score = float(llm.invoke(prompt).strip())
        return min(score / 10, 1.0)
    except:
        return 0.5

def answer_relevancy(question: str, answer: str) -> float:
    """How relevant is answer to question (0-1)"""
    prompt = f"""
    QUESTION: {question}
    ANSWER: {answer}
    
    Score 1-10: How directly does answer address the question? 
    Just give number:"""
    
    try:
        score = float(llm.invoke(prompt).strip())
        return min(score / 10, 1.0)
    except:
        return 0.5

def evaluate_rag(test_cases: List[Dict], predictions: List[Dict]) -> Dict[str, float]:
    """Full RAG evaluation"""
    results = []
    
    for tc, pred in zip(test_cases, predictions):
        contexts = pred.get('contexts', [])
        score_cp = context_precision(contexts, pred['answer'])
        score_f = faithfulness(pred['answer'], contexts)
        score_ar = answer_relevancy(tc['question'], pred['answer'])
        
        results.append({
            'context_precision': score_cp,
            'faithfulness': score_f,
            'answer_relevancy': score_ar
        })
    
    # Average scores
    avg_scores = {k: np.mean([r[k] for r in results]) for k in results[0]}
    avg_scores['overall'] = np.mean(list(avg_scores.values()))
    
    return avg_scores

# 🩺 Medical test cases
MEDICAL_TEST_CASES = [
    {
        'question': 'What are symptoms of type 2 diabetes?',
        'ground_truth': 'thirst, frequent urination, fatigue, blurred vision'
    },
    {
        'question': 'Allergy symptoms involving skin?',
        'ground_truth': 'rash, hives, itching, swelling'
    },
    {
        'question': 'Signs of heart attack in women?',
        'ground_truth': 'jaw pain, nausea, fatigue, shortness of breath'
    }
]

def run_quick_eval(ask_medx_func, num_samples=2):
    """Quick evaluation wrapper"""
    print("🧪 Running Open Source RAGAS Evaluation...")
    
    test_cases = MEDICAL_TEST_CASES[:num_samples]
    predictions = []
    
    for tc in test_cases:
        answer, contexts = ask_medx_func(tc['question'], return_contexts=True)
        # Extract contexts from your hybrid retriever (modify ask_medx to return them)
        predictions.append({
            'question': tc['question'],
            'answer': answer,
            'contexts': contexts  
        })
    
    results = evaluate_rag(test_cases, predictions)
    
    print("\n📊 EVALUATION RESULTS:")
    print(f"  Context Precision:  {results['context_precision']:.3f}")
    print(f"  Faithfulness:       {results['faithfulness']:.3f}") 
    print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
    print(f"  Overall Score:      {results['overall']:.3f}")
    
    return results
