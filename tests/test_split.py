import logging
import unittest

import polars as pl

from src.factor_calc import FactorCalculator
from src.model_train import ModelTrainer


class HoldoutTest(unittest.TestCase):
    def test_label_end_includes_other_stocks_in_the_date_group(self):
        calculator = FactorCalculator.__new__(FactorCalculator)
        calculator.logger = logging.getLogger(__name__)
        data = pl.DataFrame({
            "symbol": ["A"] * 7 + ["B"] * 7,
            "date": list(range(1, 8)) + [1, 2, 3, 4, 5, 6, 9],
            "open": [10.0] * 14,
        })
        result = calculator.compute_5d_excess_label(data)
        self.assertEqual(
            result.filter(pl.col("date") == 1)["label_end_date"].to_list(),
            [9, 9],
        )

    def test_excludes_labels_that_end_on_or_after_split_date(self):
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.logger = logging.getLogger(__name__)
        trainer.split_date = "2026-01-10"
        data = pl.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-10"],
            "label_end_date": ["2026-01-09", "2026-01-10", "2026-01-12", "2026-01-20"],
        })
        train, holdout = trainer.split_train_test(data)
        self.assertEqual(train["date"].to_list(), ["2026-01-01"])
        self.assertEqual(holdout["date"].to_list(), ["2026-01-10"])
