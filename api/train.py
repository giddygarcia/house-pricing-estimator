import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("dataset/house_listings_cleaned.csv")

# Target variable and features
X = df.drop(["Price", "PricePerSqm", "EnergyCertificate"], axis=1)
y = df["Price"]

# Splitting the data - 70/12.5/12.5
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Logging target variable
y_train_log = np.log1p(y_train)

# Preprocessor
num_features = X.select_dtypes(include=["number"]).columns
corr_nums = ["NumberOfBathrooms", "TotalRooms", "NumberOfBedrooms"]
other_nums = [col for col in num_features if col not in corr_nums]
cat_features = ["District", "City", "Town"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Missing"),
                    ),
                    (
                        "encoder",
                        TargetEncoder(
                            smooth="auto", target_type="continuous", random_state=42
                        ),
                    ),
                ]
            ),
            cat_features,
        ),
        (
            "corr_num",
            Pipeline(
                steps=[("imputer", IterativeImputer(max_iter=10, random_state=42))]
            ),
            corr_nums,
        ),
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            other_nums,
        ),
    ],
    remainder="drop",
)
preprocessor.set_output(transform="pandas")

# Full pipeline for LGBM model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=1000,
                max_depth=-1,
                learning_rate=0.03,
                num_leaves=126,
                subsample=0.8609073483994067,
                colsample_bytree=0.8,
                min_child_samples=5,
                reg_lambda=2,
                n_jobs=-1,
                verbose=-1,
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train_log)


# Evaluation
val_preds = np.expm1(pipeline.predict(X_val))
mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MAE: €{mae:,.0f}")

test_preds = np.expm1(pipeline.predict(X_test))
mae_test = mean_absolute_error(y_test, test_preds)
print(f"Test MAE: €{mae_test:,.0f}")


# Save
joblib.dump(pipeline, "lgbm_model.pkl")
print("🖨️ Model saved to lgbm_model.pkl in api folder")
