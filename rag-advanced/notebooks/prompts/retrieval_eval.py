
RETRIEVAL_EVAL_PROMPT ="""Given a query and a document excerpt, you must provide a score on an integer scale of 0 to 2 with the following meanings:
    0 = represents that the excerpt is irrelevant to the query,
    1 = represents that the excerpt is somewhat relevant to the query,
    2 = represents that the excerpt is is highly relevant to the query.
    

Important Instruction: Assign category 1 if the excerpt is somewhat related to the query but not completely, category 2 if the excerpt only and entirely refers to the query. If neither of these criteria satisfies the query, give it category 0.


Split this problem into steps:
Consider the underlying intent of the query. Measure how well the content matches a likely intent of the query(M).
Measure how trustworthy the excerpt is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). 
Final score must be an integer value only.
Do not provide any code in result. Provide each score in the following JSON format: 
{{"final_score": <integer score without providing any reasoning.>}}

## Examples

Example 1: 
<Query>
How do I programmatically access the human-readable run name?
</Query>
<Document>
If you do not explicitly name your run, a random run name will be assigned to the run to help identify the run in the UI. For instance, random run names will look like "pleasant-flower-4" or "misunderstood-glade-2".

If you'd like to overwrite the run name (like snowy-owl-10) with the run ID (like qvlp96vk) you can use this snippet:

import wandbRetrieval_Evaluation

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()

</Document>
{{"final_score": 0}}

Example 2:
<Query>
What are Runs?
</Query>
<Document>
A single unit of computation logged by W&B is called a run. You can think of a W&B run as an atomic element of your whole project. You should initiate a new run when you:
 - Train a model
 - Change a hyperparameter
 - Use a different model
 - Log data or a model as a W&B Artifact
 - Download a W&B Artifact

For example, during a sweep, W&B explores a hyperparameter search space that you specify. Each new hyperparameter combination created by the sweep is implemented and recorded as a unique run. 
</Document>
{{"final_score": 2}}

Example 3:
<Query>
How do I use W&B with Keras ?
</Query>
<Document>
We have added three new callbacks for Keras and TensorFlow users, available from wandb v0.13.4. For the legacy WandbCallback scroll down.
These new callbacks,
 - Adhere to Keras design philosophy
 - Reduce the cognitive load of using a single callback (WandbCallback) for everything
 - Make it easy for Keras users to modify the callback by subclassing it to support their niche use case
</Document>
{{"final_score": 1}}

<Query>
{query}
</Query>

<Document>
{document}
</Document>
"""
