"""
main.py
-------
Convenience entry point — run this to start RIME.

    python main.py

Or with uvicorn directly:

    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""

import uvicorn
from config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "api.main:app",
        host=s.app.host,
        port=s.app.port,
        reload=False,
        log_level=s.app.log_level.lower(),
    )
