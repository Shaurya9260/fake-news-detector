import unittest
from pathlib import Path

import psycopg
from streamlit.testing.v1 import AppTest

from database import initialize_database, recent_predictions
from utils import clean_text


APP = str(Path(__file__).resolve().parents[1] / "app.py")
COUNCIL_ARTICLE = (
    "The city council approved the budget on Wednesday after a public meeting. "
    "The plan funds a new library and expands bus service next year."
)
RUMOR_ARTICLE = (
    "Share this shocking secret now! They are hiding the miracle cure from everyone. "
    "Click immediately before the truth disappears from the internet forever."
)


def prediction_count():
    with psycopg.connect(connect_timeout=5) as connection:
        return connection.execute("SELECT count(*) FROM predictions").fetchone()[0]


class PracticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_database()

    def test_preprocessing_uses_original_english_rules(self):
        self.assertEqual(clean_text("(Reuters) The cars and buses! https://example.com 2026"), "car bus")

    def test_blank_submission_adds_no_row(self):
        before = prediction_count()
        app = AppTest.from_file(APP, default_timeout=60).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Please enter some text", app.error[0].value)
        self.assertEqual(prediction_count(), before)

    def test_model_saves_both_labels_and_history_survives_a_new_session(self):
        app = AppTest.from_file(APP, default_timeout=60).run()
        before = prediction_count()
        for article, label in [(COUNCIL_ARTICLE, "REAL"), (RUMOR_ARTICLE, "FAKE")]:
            with self.subTest(label=label):
                app.text_area[0].set_value(article)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                row = recent_predictions(1)[0]
                self.assertEqual(row["article_preview"], article)
                self.assertEqual(row["label"], label)
                self.assertGreaterEqual(row["confidence"], 50)
                self.assertLessEqual(row["confidence"], 100)
                metric = app.metric[0 if label == "FAKE" else 1]
                self.assertEqual(metric.value, f'{row["confidence"]:.1f}%')

        fresh_app = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(fresh_app.exception)
        self.assertEqual(fresh_app.dataframe[0].value["label"].iloc[:2].tolist(), ["FAKE", "REAL"])
        self.assertEqual(prediction_count(), before + 2)
