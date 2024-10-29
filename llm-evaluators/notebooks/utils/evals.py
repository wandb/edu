import pandas as pd
import weave

def get_evaluation_predictions(weave_client, eval_call_id: str) -> pd.DataFrame:
    """
    Retrieves evaluation predictions from a Weave call and returns them as a DataFrame.
    
    Args:
        eval_call_id (str): The ID of the Weave evaluation call to analyze
        
    Returns:
        pd.DataFrame: DataFrame containing the evaluation data with predictions
    """
    eval_calls = weave_client.get_call(eval_call_id)

    predictions = []
    for eval_call in eval_calls.children():
        if eval_call.op_name.split("/")[-1].split(":")[0] == "Evaluation.predict_and_score":
            _eval_call = weave_client.get_call(eval_call.id)

            # data = dict(_eval_call.inputs["example"])
            # data.update({"pred_score": dict(_eval_call.output)["model_output"]["score"]})
            # predictions.append(data)

    return pd.DataFrame(predictions)