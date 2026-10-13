
import sys as _sys

# Windows consoles and pipes default to cp1252: a stray "−" or "✓" from a tutor
# reply or the journal would crash a print. Prefer UTF-8, never crash.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
