"""Pause this JarvisLabs instance via the API (jarvislabs SDK in /home/envs/env-jl).

  /home/envs/env-jl/bin/python scripts/jl_pause.py --check   # read-only: this instance's id, GPU, status
  /home/envs/env-jl/bin/python scripts/jl_pause.py --pause

The token is read from /home/.jarvislabs_token and never printed, logged or written anywhere; any error message is
scrubbed of it before printing. The instance id comes from $MACHINE_ID (set by JarvisLabs) unless --machine-id is given.
"""
import argparse
import os
import sys

TOKEN_FILE = "/home/.jarvislabs_token"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--pause", action="store_true")
    ap.add_argument("--machine-id", type=int, default=None)
    args = ap.parse_args()
    token = open(TOKEN_FILE).read().strip()
    scrub = lambda s: str(s).replace(token, "<token>")  # noqa: E731
    mid = args.machine_id if args.machine_id is not None else int(os.environ["MACHINE_ID"])
    try:
        from jarvislabs import Client
        with Client(api_key=token) as c:
            inst = c.instances.get(mid)
            print(f"instance {inst.machine_id}: gpu={inst.gpu_type} status={inst.status} name={inst.name}", flush=True)
            if args.pause:
                c.instances.pause(mid)
                print(f"PAUSE_REQUESTED {mid}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"JL_API_ERROR {type(e).__name__}: {scrub(e)}", file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
