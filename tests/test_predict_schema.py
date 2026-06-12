from src.predict import format_prediction


def test_format_prediction_matches_label_schema():
    rec = format_prediction(1024.5, 680.2, "L-Shape")
    assert set(rec.keys()) == {"mark", "verified_shape"}
    assert set(rec["mark"].keys()) == {"x", "y"}
    assert rec["verified_shape"] == "L-Shape"
    assert isinstance(rec["mark"]["x"], float)
