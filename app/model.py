import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from app.config import config
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelWrapper:
    def __init__(self, synthetic_samples: int = None):
        self.synthetic_samples = synthetic_samples or config.SYNTHETIC_SAMPLES
        self.model: Pipeline = None
        self._build_and_train()

    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        df = pd.DataFrame({
            "brand": np.random.choice(["Dell", "HP", "Lenovo", "Asus", "Acer"], size=n_samples),
            "ram": np.random.randint(4, 64, size=n_samples),
            "storage": np.random.randint(128, 2000, size=n_samples),
            "weight": np.random.uniform(1.0, 3.5, size=n_samples),
            "cpu_freq_ghz": np.round(np.random.uniform(1.0, 4.5, size=n_samples), 2)
        })

        df["price_usd"] = (
            200 +
            df["ram"] * 8 +
            df["storage"] / 128 * 12 +
            df["cpu_freq_ghz"] * 40 -
            df["weight"] * 20 +
            np.random.normal(0, 50, n_samples)
        ).clip(100, None)

        return df

    def _build_pipeline(self) -> Pipeline:
        categorical_cols = ["brand"]
        numeric_cols = ["ram", "storage", "weight", "cpu_freq_ghz"]

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ])

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0, random_state=config.RANDOM_SEED))
        ])

        return pipeline

    def _build_and_train(self):
        df = self._generate_synthetic_data(self.synthetic_samples)
        X = df.drop(columns=["price_usd"])
        y = df["price_usd"]

        pipeline = self._build_pipeline()
        pipeline.fit(X, y)

        self.model = pipeline
        logger.info(f"Model trained on {len(X)} samples")

    def predict(self, features: dict) -> float:
        expected_cols = ["brand", "ram", "storage", "weight", "cpu_freq_ghz"]
        for col in expected_cols:
            if col not in features:
                raise ValueError(f"Missing feature: {col}")

        df = pd.DataFrame([features])
        pred = self.model.predict(df)
        return float(pred[0])


# Global instance
model_wrapper = ModelWrapper()
