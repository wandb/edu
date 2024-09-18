<p align="center">
  <img src="https://raw.githubusercontent.com/wandb/wandb/508982e50e82c54cbf0dd464a9959fee0e1740ad/.github/wb-logo-lightbg.png#gh-light-mode-only" width="600" alt="Weights & Biases"/>
  <img src="https://raw.githubusercontent.com/wandb/wandb/508982e50e82c54cbf0dd464a9959fee0e1740ad/.github/wb-logo-darkbg.png#gh-dark-mode-only" width="600" alt="Weights & Biases"/>
</p>

# Machine Learning for Business Decision Optimization

This repository contains materials for our [Machine Learning for Business Decision Optimization](https://www.wandb.courses/courses/decision-optimization) course. 

Learn to optimize decision rules, translating machine learning predictions into actionable insights. Discover how to achieve practical value and business impact by measuring performance using business metrics, and deploy ML models successfully.

## 🚀 [Enroll for free](https://www.wandb.courses/courses/decision-optimization)

## What you'll learn

### Translate Machine Learning Predictions Into Actionable Business Insights
Learn how to identify decision and prediction problems in your work and optimize them using advanced machine learning models. Understand the significance of transforming abstract data into direct, actionable insights that can power your business decisions and strategies.

### Master Business Impact Measurement for ML Models
Instead of relying solely on accuracy metrics, you'll master how to measure the performance of your ML models in dollar terms. Discover the power of using profit curves for optimizing binary decisions, and uncover how to address their basic limitations. This unique approach will enhance your ability to drive impactful results and high return on investment from your ML initiatives.

### Choose the Right Loss Function for Your Business Problem
Enhance your proficiency in selecting the most appropriate loss function for your specific business problem. This crucial skill will empower you to implement ML models more effectively, ensuring they are attuned to your business objectives, thereby increasing their relevance and impact.

## Running the code

- Notebooks can be run on your local system or via Google Colab
- If you have questions, you can ask them in [Discord](https://wandb.me/discord) in the `#courses` channel

## Notebooks

- 1_profit_curves.ipynb: Notebook used in the first lesson
- 2_theory.ipynb: Notebook showing the behavior of different loss functions
- 2_bimbo.ipynb: Applying decision optimization for regression.
- 3_dynamic_opt_data_prep.ipynb: This creates the saved models and artifacts used in `3_dynamic_decision_opt.ipynb`. This calls `./utils/modeling.py`
- 3_dynamic_decision_opt.ipynb: This repository doesn't contain all of the saved model files. So run `3_dynamic_opt_data_prep.ipynb` before this.
