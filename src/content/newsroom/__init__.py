"""BachTalk newsroom: a multi-agent editorial team that turns Savant/FanGraphs
data into Barstool-style baseball threads.

Pipeline:  feeds (wire) -> researcher (fact sheet) -> writer (columnist)
           -> social (thread packager) -> NewsroomGenerator (orchestrator)

Additive: this package does not modify any existing generator. It plugs into the
scheduler as one new generator (`newsroom`).
"""

from .newsroom import NewsroomGenerator

__all__ = ["NewsroomGenerator"]
