def _is_licence_403(err_text: str) -> bool:
    """Helper to detect CDS licence errors deterministically."""
    txt = err_text.lower()
    return "required licences not accepted" in txt

def test_licence_detection_logic():
    # Simulate a CDS 403 error message
    err_msg = "Client Error: 403 Forbidden: Required licences not accepted. Please check..."
    assert _is_licence_403(err_msg) is True

    # Simulate generic 403
    err_msg_2 = "Client Error: 403 Forbidden: Invalid credentials"
    assert _is_licence_403(err_msg_2) is False

    # Simulate 404
    err_msg_3 = "Client Error: 404 Not Found"
    assert _is_licence_403(err_msg_3) is False
