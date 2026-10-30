"""Run inside QuantRocket Jupyter after the user activates their free license.

This creates/ingests only the daily 2007-2011 learning bundle. It never sets a
license key, buys data, connects a broker, starts an algorithm, or exports raw data.
An asynchronous response is NOT proof of completed ingestion. Check zipline logs
for completion and audit the ingested data before fitting any forecasting model.
"""

import json

from quantrocket.zipline import create_usstock_bundle, ingest_bundle, list_bundles

BUNDLE = "usstock-learn-1d"


def main():
    existing = list_bundles()
    if BUNDLE not in existing:
        created = create_usstock_bundle(BUNDLE, learn=True)
        print(json.dumps({"bundle_created": created}, default=str))
    else:
        raise RuntimeError(
            "The learning bundle already exists. Inspect its status/logs before deciding whether to re-ingest; "
            "this script will not replace or re-ingest existing data automatically."
        )
    submitted = ingest_bundle(BUNDLE)
    print(json.dumps({"ingestion_requested": submitted, "completed": False}, default=str))
    print("Monitor zipline service logs; verify this bundle's completion message and data coverage.")
    print(json.dumps({"bundle_status_snapshot": list_bundles()}, default=str))


if __name__ == "__main__":
    main()
