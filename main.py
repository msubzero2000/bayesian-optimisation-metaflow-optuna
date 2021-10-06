"""
Optuna example that optimizes a classifier configuration for cancer dataset using
Catboost.

In this example, we optimize the validation accuracy of cancer detection using
Catboost. We optimize both the choice of booster model and their hyperparameters.

"""
import logging
from datetime import datetime, timedelta

from metaflow import FlowSpec, step, S3, current, resources, conda_base, Parameter, batch, project

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

@conda_base(libraries={'catboost': '0.26.1',
                       'optuna': '2.9.1',
                       'scikit-learn': '1.0',
                       'sqlalchemy': '1.4.22',
                       'psycopg2-binary':'2.9.1'})
class Optimisation(FlowSpec):

    def objective(self, trial):
        import optuna
        import numpy as np
        import catboost as cb
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        data, target = load_breast_cancer(return_X_y=True)
        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        gbm = cb.CatBoostClassifier(**param)

        gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

        preds = gbm.predict(valid_x)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(valid_y, pred_labels)

        return accuracy

    @batch(cpu=4)
    @step
    def start(self):
        import optuna
        from distributed_optuna import DistributedOptuna, DistributedOptunaInfo, DistributedOptunaWorkerInfo
        info = DistributedOptuna.create(num_workers=10, num_trials=10, timeout=600, study_name_prefix="catboost")
        print(f"Creating study {info.study_name} with {len(info.workers_info)} workers")
        study = optuna.create_study(direction="maximize", study_name=info.study_name, storage=info.storage)

        self.optimisation_params = info.workers_info
        self.next(self.optimise, foreach="optimisation_params")

    @batch(cpu=4)
    @step
    def optimise(self):
        import optuna
        from distributed_optuna import DistributedOptuna, DistributedOptunaInfo, DistributedOptunaWorkerInfo
        study = optuna.load_study(study_name=self.input.study_name, storage=self.input.storage)
        study.optimize(self.objective, n_trials=self.input.num_trials, timeout=self.input.timeout)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Record Best trial:")
        # Record this worker's best trial
        self.trial = study.best_trial

        self.next(self.join)

    @step
    def join(self, inputs):
        # Find the global best trial amongst worker's best trial
        best_trial = None
        best_trial_value = 0

        for input in inputs:
            cur_best_trial = input.trial
            print(f"Best trial for this worker {cur_best_trial.value}")

            if best_trial is None or cur_best_trial.value > best_trial_value:
                best_trial = cur_best_trial
                best_trial_value = cur_best_trial.value

        print("Value: {}".format(best_trial.value))

        print("Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        self.next(self.end)

    @step
    def end(self):
        print("Completed")


opt = Optimisation()
