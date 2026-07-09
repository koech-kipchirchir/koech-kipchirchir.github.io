#!/usr/bin/env python3
"""
AIOS Health Check Script
Used by Docker HEALTHCHECK and external monitoring systems.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def check(url: str, timeout: int = 10) -> None:
    try:
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode()
        status = resp.getcode()

        if status == 200:
            try:
                data = json.loads(body)
                if data.get("status") == "ok" or data.get("status") == "healthy":
                    print(f"OK: {url} returned 200 (healthy)")
                    sys.exit(0)
                else:
                    print(f"UNHEALTHY: {url} returned unexpected status: {data}")
                    sys.exit(1)
            except json.JSONDecodeError:
                print(f"OK: {url} returned 200")
                sys.exit(0)
        else:
            print(f"UNHEALTHY: {url} returned {status}")
            sys.exit(1)

    except urllib.error.HTTPError as e:
        if e.code == 503:
            print(f"NOT_READY: {url} returned 503 (not ready)")
            sys.exit(1)
        print(f"ERROR: {url} returned {e.code}")
        sys.exit(1)

    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach {url}: {e.reason}")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOS health check")
    parser.add_argument("--url", default="http://localhost:8000/v1/live",
                        help="Health endpoint URL")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Request timeout in seconds")
    args = parser.parse_args()
    check(args.url, args.timeout)
