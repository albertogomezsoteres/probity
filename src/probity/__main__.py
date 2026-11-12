"""Allow ``python -m probity``."""

import sys

from .cli import main

sys.exit(main())
