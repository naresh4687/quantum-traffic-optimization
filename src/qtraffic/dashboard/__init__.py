"""Control-room dashboard: presentation and control layer over the validated qtraffic system.

Pure logic lives in ``state``, ``saved``, ``theme``, ``charts`` and ``network_view`` (no Streamlit needed); only
``components`` and the top-level ``app.py`` import Streamlit.
"""
