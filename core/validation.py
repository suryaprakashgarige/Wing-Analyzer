def validate_results(CL, CD):
    """
    Validates aggregate wing results against physical reality.
    """
    if not (0 < CL < 2):
        raise ValueError(f"Invalid CL: {CL:.3f}. Expected range (0, 2).")

    if not (0 < CD < 0.2):
        raise ValueError(f"Invalid CD: {CD:.4f}. Expected range (0, 0.2).")

    LD = CL / CD
    if not (5 < LD < 30):
        # We raise a warning or value error if it's physiologically impossible
        raise ValueError(f"Unrealistic L/D: {LD:.2f}. Expected range (5, 30).")

    return True
