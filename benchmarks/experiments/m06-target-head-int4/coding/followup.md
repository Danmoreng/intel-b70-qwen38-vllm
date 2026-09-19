
Make the Tasks page accurately distinguish execution disabled by configuration
from an unreachable configured runner. The current generic safety-check wording
is misleading when no runner client was constructed. Trace the backend status
through contracts and UI; present useful English and German text consistent
with each actual condition. Do not infer completed safety checks or runner health
from an open event stream. Add relevant regression tests, retain existing API
compatibility where practical, and run the final quality checks.

