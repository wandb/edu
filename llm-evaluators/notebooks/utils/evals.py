import pandas as pd
from sklearn.metrics import cohen_kappa_score

def get_evaluation_predictions(weave_client, eval_call_id):
    """
    Extract and format evaluation predictions from a Weave evaluation call ID.
    
    Args:
        eval_call_id (str): ID of the Weave evaluation call to analyze
        
    Returns:
        pd.DataFrame: DataFrame containing paired human and model scores for each metric
    """
    eval_calls = weave_client.get_call(eval_call_id)
    predictions = []
    
    for eval_call in eval_calls.children():
        if eval_call.op_name.split("/")[-1].split(":")[0] == "Evaluation.predict_and_score":
            _eval_call = weave_client.get_call(eval_call.id)
            
            # Extract data
            input_text = _eval_call.inputs["example"]["input"]
            human_scores = _eval_call.inputs["example"]["scores"]
            model_scores = _eval_call.output["scores"]
            
            # Create paired scores
            scores = {
                'input': input_text,
                'required_keys': (human_scores['human_required_keys'], model_scores['test_adheres_to_required_keys']),
                'word_limit': (human_scores['human_word_limit'], model_scores['test_adheres_to_word_limit']),
                'privacy': (human_scores['human_absence_of_PII'], model_scores['judge_adheres_to_privacy_guidelines']),
                'overall': (human_scores['human_overall_score'], model_scores['judge_overall_score'])
            }
            predictions.append(scores)

    return pd.DataFrame(predictions)

def calculate_kappa_scores(df, tuple_columns=['required_keys', 'word_limit', 'privacy', 'overall']):
    """
    Calculate Cohen's Kappa scores for human vs model predictions across multiple metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing paired scores as tuples (human_score, model_score)
        tuple_columns (list): List of column names containing the score tuples
        
    Returns:
        dict: Dictionary of kappa scores for each metric
    """
    labels = [0, 1]  # Binary classification labels
    kappa_scores = {}
    
    for col in tuple_columns:
        human_scores = df[col].apply(lambda x: x[0])
        pred_scores = df[col].apply(lambda x: x[1])
        
        kappa_scores[col] = cohen_kappa_score(
            human_scores,
            pred_scores,
            labels=labels,
            weights='linear'
        )
    
    return kappa_scores

def calculate_weighted_alignment(kappa_scores, weights=None):
    """
    Calculate weighted alignment score across all metrics.
    
    Args:
        kappa_scores (dict): Dictionary of kappa scores for each metric
        weights (dict): Optional dictionary of weights for each metric. 
                       If None, uses equal weights.
    
    Returns:
        float: Weighted average kappa score
    """
    # Default to equal weights if none provided
    if weights is None:
        weights = {metric: 1/len(kappa_scores) for metric in kappa_scores.keys()}
    
    # Validate weights
    assert set(weights.keys()) == set(kappa_scores.keys()), \
        "Weights must be provided for all metrics"
    assert abs(sum(weights.values()) - 1.0) < 1e-9, \
        "Weights must sum to 1"
    
    # Calculate weighted average
    weighted_score = sum(kappa_scores[metric] * weights[metric] 
                        for metric in kappa_scores.keys())
    
    return weighted_score
