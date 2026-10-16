"""Unit and integration tests for the Streamlit web application (app.py)."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_app_initial_load():
    """Verify that app.py loads cleanly without exceptions."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    assert "Fastly CyberNode" in at.title[0].value
    assert len(at.file_uploader) == 1


def test_app_in_memory_file_upload_evaluation():
    """Verify in-memory CSV file upload and rule evaluation workflow."""
    app_path = Path(__file__).parent.parent / "app.py"
    sample_csv = Path(__file__).parent.parent / "requests_sqli_xss_cmdexe.csv"
    assert sample_csv.exists(), "requests_sqli_xss_cmdexe.csv must exist"

    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()

    # Upload CSV in memory
    at.file_uploader[0].upload("requests_sqli_xss_cmdexe.csv", sample_csv.read_bytes())
    at.run()
    assert not at.exception

    # Select row 0
    at.number_input[0].set_value(0)
    at.run()
    assert not at.exception

    # Click run button
    run_btn = [b for b in at.button if "Analyze Log & Recommend Rule" in b.label][0]
    run_btn.click()
    at.run()
    assert not at.exception

    # Verify rule match results
    success_texts = [s.value for s in at.success]
    assert any("RULE-002" in text for text in success_texts), f"Expected RULE-002 in {success_texts}"


def test_app_sample_dataset_selection():
    """Verify selecting a bundled sample dataset from sidebar."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()

    # Select sample from sidebar
    at.sidebar.selectbox[0].select("requests_maliciousip.csv")
    at.run()
    assert not at.exception

    # Verify metrics updated
    metric_labels = [m.label for m in at.metric]
    assert "Total Log Rows" in metric_labels

    # Run analysis on row 0
    run_btn = [b for b in at.button if "Analyze Log & Recommend Rule" in b.label][0]
    run_btn.click()
    at.run()
    assert not at.exception

    success_texts = [s.value for s in at.success]
    assert any("RULE-001" in text for text in success_texts), f"Expected RULE-001 in {success_texts}"
