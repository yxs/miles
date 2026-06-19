"""Omni RL rollout integration for the sglang-omni inference backend.

This package keeps omni-specific rollout glue out of generic miles core. It is loaded
through path-string hooks (``--custom-generate-function-path``) and imports only public
miles entrypoints so it can later be extracted into a standalone distribution.
"""
