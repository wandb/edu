import catboost
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics


class CatBoostTrainer(object):
    def __init__(self, hparams: dict):
        self.hparams = hparams
        if "random_seed" not in self.hparams:
            self.hparams["random_seed"] = 42
        if "loss_function" not in self.hparams:
            self.hparams["loss_function"] = "RMSE"
        if "eval_metric" not in self.hparams:
            self.hparams["eval_metric"] = "RMSE"

        self.model = catboost.CatBoostRegressor(**self.hparams)

    def fit(self, X, y, verbose=False, callbacks=None):
        # Split X and y into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        self.model.fit(
            X_train,
            y_train,
            verbose=verbose,
            eval_set=(X_test, y_test),
            callbacks=callbacks,
        )

    def predict(self, X):
        preds = self.model.predict(X)
        return preds
        # if preds.ndim == 2:
        #     return preds[:, 0]

    def score(self, X, y, metric: str = "MSE"):
        if metric == "MSE":
            return metrics.mean_squared_error(y, self.predict(X))

        raise ValueError(f"Metric {metric} not supported.")

    def get_feature_importance(self):
        importances = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values(by="importance", ascending=False)
