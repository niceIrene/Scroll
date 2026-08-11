import scroll_eval


def test_package_has_version() -> None:
    assert scroll_eval.__version__ == "0.1.0"
