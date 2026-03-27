import dgd


def pytest_report_header(config):
    """Reports the imported dgd module path in the pytest session header."""
    del config
    return [f"dgd module: {dgd.__file__}"]
