# Legacy facial recognition smart lock (2022)

This directory archives a recovered Raspberry Pi university prototype, preserved as a baseline for the modern redevelopment. The original implementation was developed with assistance.

## Contents
- `smartlock.py`: Tkinter entry point and application controller
- `view.py`: enrolment, facial embeddings, identification, camera interface and GPIO actuation
- `config.py` and `config.ini`: original configuration code and settings

## Security and reproducibility

**Historical reference only — not safe for physical access control.**
The original `if names:` check will activate the lock for *any detected face*, including unknown faces. Do not operate a physical lock using this unmodified implementation.

The original facial images, biometric embeddings, logs, and Python caches are deliberately excluded from the public repository. A personal identifier in the header of `view.py` is redacted for privacy. No compatibility or accuracy claims have been validated against current hardware.

The archived implementation may incorporate assisted or third-party contributions: confirm redistribution rights before publicly licensing those contributions.
