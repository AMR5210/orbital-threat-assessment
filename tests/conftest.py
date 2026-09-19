"""Loads .env before any test runs, so ANTHROPIC_API_KEY / SNOWFLAKE_*
credential-gated tests see them the same way the standalone scripts do
(see src/otda/*.py and scripts/*.py, which each call load_dotenv()
directly — pytest has no equivalent built-in, hence this file)."""
from dotenv import load_dotenv

load_dotenv()
