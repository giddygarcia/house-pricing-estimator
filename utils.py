from typing import Callable
import joblib
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
from optuna.pruners import MedianPruner
from scipy.stats import kruskal
from sklearn.base import TransformerMixin, clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

search_spaces = {
    "Random Forest": lambda trial: {
        "criterion": "absolute_error",
        "n_estimators": trial.suggest_int("n_estimators", 200, 280),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        "max_features": trial.suggest_float("max_features", 0.6, 0.8),
        "max_samples": 0.8,
        "n_jobs": -1,
    },
    "Light Gradient Boosting": lambda trial: {
        "objective": "regression_l1",
        "n_estimators": 1000,
        "max_depth": trial.suggest_categorical("max_depth", [*range(5, 13), -1]),
        "learning_rate": 0.03,
        "num_leaves": trial.suggest_int("num_leaves", 100, 150),
        "subsample": trial.suggest_float("subsample", 0.8, 0.9),
        "colsample_bytree": 0.8,
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 5),
        "reg_lambda": 2,
        "n_jobs": -1,
        "verbose": -1,
    },
    "Extreme Gradient Boosting": lambda trial: {
        "objective": "reg:absoluteerror",
        "n_estimators": 1000,
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.8, 0.9),
        "colsample_bytree": 0.6,
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 5),
        "reg_lambda": 0.5,
        "gamma": 1,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    },
}


def fmt_price_axis(
    ax, cols=("Price", "PricePerSqm", "MAE", "RMSE"), force_x=False, force_y=False
) -> None:
    """
    Format plot axes with Euro currency labels where applicable.

    Format either x/y label if axes label matches specified column names of currency, or if non-matching use force label to override and apply formatting.

    Args:
        ax: the matplotlib axes to format.
        cols: names of currency columns to edit.
        force_x: Used to format non-specified columns on x-axis.
        force_y: Used to format non-specified columns on y-axis.

    Returns: None
    """
    euro_fmt = plt.FuncFormatter(
        lambda x, _: f"€{x / 1e6:.1f}M" if x >= 1e6 else f"€{x:,.0f}"
    )
    if force_x or ax.get_xlabel() in cols:
        ax.xaxis.set_major_formatter(euro_fmt)
    if force_y or ax.get_ylabel() in cols:
        ax.yaxis.set_major_formatter(euro_fmt)


def score_predictions(
    y_pred_train: pd.Series,
    y_pred_val: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    model_name: str,
    params: dict,
    results: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score model predictions across multiple regression metrics and append results to a tracked df.

    Args:
        y_pred_train: Predicted values for the training set.
        y_pred_val: Predicted values for the validation set.
        y_train: True target values for the training set.
        y_val: True target values for the validation set.
        model_name: Label name to identify the model in the dataframe.
        params: Model hyperparameters or configuration, stored as a string.
        results: Existing results DataFrame to append to. If ``None``, a new
            empty DataFrame is created.

    Returns:
        results: unformatted dataframe of results with all scores appended.
        new_results: formatted copy of results, sorted ascending with currency formatting applied where applicable.
    """
    if results is None:
        results = pd.DataFrame(
            columns=["Model", "MAE", "Train MAE", "RMSE", "R2", "Params"]
        )

    row = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_val, y_pred_val),
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "RMSE": root_mean_squared_error(y_val, y_pred_val),
        "R2": r2_score(y_val, y_pred_val),
        "Params": str(params),
    }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    metric_cols = ["MAE", "Train MAE", "RMSE", "R2"]
    new_results = results.sort_values(by="MAE").reset_index(drop=True)
    new_results[metric_cols] = new_results[metric_cols].astype(float).round(3)

    for col in metric_cols[:3]:
        new_results[col] = new_results[col].apply(lambda x: f"€{x:,.3f}")

    return results, new_results


def objective(
    trial: optuna.Trial,
    model_class: type,
    param_func: Callable[[optuna.Trial], dict],
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    y_train: pd.Series,
    preprocessor: TransformerMixin,
) -> float:
    """
    Optuna objective function for hyperparameter tuning varying types of regression models with cross-validation.

    Performs 3-fold cross-validation, training in log space and evaluating MAE in original
    currency scale.
    Supports early stopping for LGBMRegressor and XGBRegressor.
    Supports pruning trials early through reporting unpromising trials.

    Args:
        trial: Optuna Trial object used to sample hyperparameters and report intermediate values.
        model_class: The regression model class to instantiate (e.g. LGBMRegressor, XGBRegressor).
        param_func: Callable that accepts a trial and returns a dict of hyperparameters to pass
                    to model_class.
        X_train: Train set data.
        y_train_log: Log-transformed target values.
        y_train: Original-scale target values (for viewing MAE in real currency scale).
        preprocessor: Fitted or unfitted sklearn preprocessor/pipeline step. Cloned and
                      fit separately on each fold to prevent data leakage.
    Raises:
        optuna.exceptions.TrialPruned: If the trial is pruned based on intermediate fold scores.
    Returns:
        Mean MAE across all folds in original currency scale.
    """
    params = param_func(trial)
    model = model_class(**params, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

    KF = KFold(n_splits=3, shuffle=True, random_state=42)
    maes = []
    best_iterations = []

    for fold_num, (train_idx, val_idx) in enumerate(KF.split(X_train)):
        p = clone(pipeline)

        X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_v = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

        X_tr_t = p["preprocessor"].fit_transform(X_tr, y_tr)
        X_v_t = p["preprocessor"].transform(X_v)

        model_name = type(p["regressor"]).__name__

        if model_name == "LGBMRegressor":
            p["regressor"].fit(
                X_tr_t,
                y_tr,
                eval_set=[(X_v_t, y_v)],
                eval_metric="l1",
                callbacks=[lgbm.early_stopping(50, verbose=False)],
            )
            best_iterations.append(p["regressor"].best_iteration_)

        elif model_name == "XGBRegressor":
            p["regressor"].fit(X_tr_t, y_tr, eval_set=[(X_v_t, y_v)], verbose=False)
            best_iterations.append(p["regressor"].best_iteration)

        else:
            p["regressor"].fit(X_tr_t, y_tr)

        y_pred = np.expm1(p["regressor"].predict(X_v_t))
        mae = mean_absolute_error(y_train.iloc[val_idx], y_pred)
        maes.append(mae)

        running_mae = float(np.mean(maes))
        trial.report(running_mae, step=fold_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if best_iterations:
        trial.set_user_attr("best_n_estimators", int(np.mean(best_iterations)))

    return float(np.mean(maes))


def early_stopping(
    study: optuna.Study,
    trial: optuna.Trial,
    patience: int = 25,
    min_delta_pct: float = 0.005,
) -> None:
    """
    Enforces early stopping through Optuna callback if no meaningful improvement in x trials is observed.

    Defaults stop optimization study if in 25 trials there has been no improvement in MAE by 0.5%.

    Args:
        study: the Optuna study being optimized.
        trial: Most recent completed trial.
        patience: Number of trials to check for improvement.
        min_delta_pct: Minimum fractional improvement to observe in order to continue.

    Returns: None
    """
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < patience:
        return

    best = study.best_value
    threshold = best * (1 + min_delta_pct)
    recent = completed[-patience:]

    if all(t.value > threshold for t in recent):
        print(
            f"🛑 Early stopping: No {min_delta_pct * 100:.1f}% improvement in {patience} trials"
        )
        study.stop()


def tune_models(
    models: dict,
    param_funcs: dict,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    y_train: pd.Series,
    preprocessor: TransformerMixin,
    n_trials: int = 50,
) -> dict:
    """
    Tune multiple regression models using Optuna with cross-validated MAE as the objective.

    Runs an Optuna study for each model sequentially.

    Args:
        models: Dictionary mapping of models to tune.
        param_funcs: Dictionary mapping of hyperparamaters per model to tune.
        X_train: Train set data.
        y_train_log: Log-transformed target values.
        y_train: Original-scale target values (for viewing MAE in real currency scale).
        preprocessor: Fitted or unfitted sklearn preprocessor/pipeline step. Cloned and
                      fit separately on each fold to prevent data leakage.
        n_trials: Maximum number of trials per model.

    Returns:
        best_params_dict: Dictionary mapping of best hyperparameters found during tuning per model.
    """
    best_params_dict = {}

    for name, model in models.items():
        try:
            study = optuna.create_study(
                direction="minimize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial, mc=type(model), pg=param_funcs[name]: objective(
                    trial, mc, pg, X_train, y_train_log, y_train, preprocessor
                ),
                n_trials=n_trials,
                n_jobs=1,
                show_progress_bar=True,
                callbacks=[
                    lambda study, trial: early_stopping(
                        study, trial, patience=25, min_delta_pct=0.005
                    )
                ],
            )

            best_params_dict[name] = study.best_params
            if name in ("Light Gradient Boosting", "Extreme Gradient Boosting"):
                best_params_dict[name]["n_estimators"] = (
                    study.best_trial.user_attrs.get("best_n_estimators", 1000)
                )
            print(f"✅ Successfully tuned {name}")

        except Exception as e:
            print(f"🛑 Failed to tune {name}: {e}")

    return best_params_dict


def make_pipeline_boilerplate(
    preprocessor: TransformerMixin, model_class: type, params: dict
) -> Pipeline:
    """
    Construct a sklearn Pipeline with model configuration. Serves as boilerplate to automatically inject random state, parallelism and silence verbose output for supported types

    Args:
        preprocessor:
        model_class: The specific model to instantiate.
        params: Dictionary of specific hyperparameters to pass tot he model.

    Returns:
        Pipeline with preprocessor (feature transformation handler) and regressor (actual model)
    """
    params = {**params}
    if "random_state" not in params:
        params["random_state"] = 42

    model = model_class(**params)

    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=-1)

    verbose_map = {
        "LGBMRegressor": {"verbose": -1},
        "XGBRegressor": {"verbosity": 0},
        "RandomForestRegressor": {"verbose": 0},
    }
    if model_class.__name__ in verbose_map:
        model.set_params(**verbose_map[model_class.__name__])

    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])
