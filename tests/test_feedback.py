import json
from learning.feedback import FeedbackCollector


def test_log_interaction(tmp_path):
    path = tmp_path / "training_examples.jsonl"
    FeedbackCollector.LOG_FILE = str(path)
    FeedbackCollector.log_interaction(
        query="question",
        schema={"tables": ["t"]},
        generated_sql="SELECT 1",
        error="error",
        corrected_sql="SELECT 1",
    )
    assert path.exists()
    data = json.loads(path.read_text().strip())
    assert data["generated_sql"] == "SELECT 1"
    assert data["error"] == "error"
