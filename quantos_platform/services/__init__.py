"""
Page services: plain functions that turn job artifacts into JSON-ready dicts for one page each.

They hold the page-level logic that used to live in the Streamlit views (labels, verdict rules, plain-language
readings), so the FastAPI service, the static export and the Streamlit research lab all show the same numbers.
Services only read artifacts; they never fetch data or run models (that is the jobs' work).

    common.py   shared wording helpers (percentile words, ordinals) and freshness lookups
    today.py    the Today brief
"""
