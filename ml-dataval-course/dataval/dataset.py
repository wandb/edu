import duckdb
import pandas as pd

import random

from dataval.train import CatBoostTrainer


class WeatherDataset(object):
    def __init__(self, dir_path: str, sample_frac=0.1):
        self.dir_path = dir_path
        self.con = duckdb.connect(":memory:")
        self.partitions = self.load_train(sample_frac=sample_frac)

    def load_train(self, sample_frac=0.1):
        df = (
            self.con.execute(
                f"SELECT * FROM read_csv_auto('{self.dir_path}/shifts_canonical_train.csv', SAMPLE_SIZE=1000, PARALLEL=TRUE)"
            )
            .fetchdf()
            .sample(frac=sample_frac, random_state=42)
            .reset_index(drop=True)
        )

        df["fact_time"] = pd.to_datetime(df["fact_time"], unit="s")
        df = df.sort_values(by=["fact_time"], ascending=True).reset_index(
            drop=True
        )
        df["year_week"] = (
            df["fact_time"]
            .dt.isocalendar()
            .year.astype(str)
            .str.cat(
                df["fact_time"]
                .dt.isocalendar()
                .week.map("{:02d}".format)
                .astype(str),
                sep="_",
            )
        )

        # Partition
        return {key: data for key, data in df.groupby("year_week")}

    def get_partition_keys(self):
        return list(self.partitions.keys())

    @staticmethod
    def get_partition_key(df):
        return df["year_week"].unique()[0]

    def load(self, key: str):
        if key not in self.partitions:
            raise ValueError(f"Key {key} not in partitions.")

        return self.partitions[key]

    def iterate(self):
        # Generate train-test pairs
        partitions = self.get_partition_keys()

        for train_key, test_key in zip(partitions[:-1], partitions[1:]):
            yield (self.partitions[train_key], self.partitions[test_key])

    @staticmethod
    def split_feature_label(df):
        X = df.iloc[:, 6:-1]
        y = df["fact_temperature"]
        return X, y

    @staticmethod
    def get_sensor_groups():
        return ["cmc", "gfs", "gfs_temperature", "wrf"]

    @staticmethod
    def corrupt_null(df, sensor_group, corruption_rate=0.1):
        sensor_cols = [col for col in df.columns if sensor_group in col]
        copy = df.copy()
        copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] = float("nan")
        return copy, sensor_cols

    @staticmethod
    def corrupt_nonnegative(df, sensor_group, corruption_rate=0.1):
        summary = df.describe()
        sensor_cols = [
            col
            for col in df.columns
            if sensor_group in col and summary[col].min() >= 0
        ]
        copy = df.copy()

        # Set to random negative value
        copy.loc[
            df.sample(frac=corruption_rate, random_state=42).index, sensor_cols
        ] = -1 * random.randrange(0, 100)
        return copy, sensor_cols

    @staticmethod
    def corrupt_typecheck(df, sensor_group, corruption_rate=0.1):
        sensor_cols = [
            col
            for col in df.columns
            if sensor_group in col and all(x.is_integer() for x in df[col])
        ]
        copy = df.copy()

        # Set to random float
        copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] = (
            random.random() * 100
        )
        return copy, sensor_cols

    @staticmethod
    def corrupt_units(df, sensor_group, corruption_rate=0.1):
        sensor_cols = [col for col in df.columns if sensor_group in col]
        copy = df.copy()

        # Change from Celsius to Fahrenheit
        copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] = (9 / 5) * copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] + 32
        return copy, sensor_cols

    @staticmethod
    def corrupt_average(df, sensor_group, corruption_rate=0.1):
        sensor_cols = [col for col in df.columns if sensor_group in col]
        copy = df.copy()

        # Set to average of the sensors
        copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] = copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ].mean(
            axis=1
        )
        return copy, sensor_cols

    @staticmethod
    def corrupt_pinned(df, sensor_group, corruption_rate=0.1, pinned_value=5):
        sensor_cols = [col for col in df.columns if sensor_group in col]
        copy = df.copy()

        # Set to pinned value
        copy.loc[
            copy.sample(frac=corruption_rate, random_state=42).index,
            sensor_cols,
        ] = pinned_value
        return copy, sensor_cols

    @staticmethod
    def iterate_corruptions(df, sensor_group, **kwargs):
        for corruption in [
            WeatherDataset.corrupt_null,
            WeatherDataset.corrupt_nonnegative,
            WeatherDataset.corrupt_typecheck,
            WeatherDataset.corrupt_units,
            WeatherDataset.corrupt_average,
            WeatherDataset.corrupt_pinned,
        ]:
            yield corruption.__name__, corruption(df, sensor_group, **kwargs)

    def train_and_test(self, train_df, test_df):
        X_train, y_train = self.split_feature_label(train_df)

        catboost_hparams = {
            "depth": 5,
            "iterations": 250,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
        }
        continual_t = CatBoostTrainer(catboost_hparams)
        continual_t.fit(X_train, y_train, verbose=False)
        print(
            f"Train MSE for partition {self.get_partition_key(train_df)}: {continual_t.score(X_train, y_train)}"
        )

        # Evaluate
        X_test, y_test = self.split_feature_label(test_df)
        print(
            f"Test MSE for partition {self.get_partition_key(test_df)}: {continual_t.score(X_test, y_test)}"
        )

        return (
            continual_t,
            continual_t.score(X_train, y_train),
            continual_t.score(X_test, y_test),
        )
