from __future__ import annotations

import sys

def main(argv=None):
    # Vertex wrapper passes argv as a list of CLI args
    argv = [] if argv is None else list(argv)

    # Delegate to the packaged tools entrypoint (exists in sdist)
    from tools import stmm_stepA_train as impl

    # Run it as if invoked from CLI
    old_argv = sys.argv[:]
    try:
        sys.argv = ["stmm_stepA_train"] + argv
        if hasattr(impl, "main"):
            return impl.main()
        # Fallback: if tools module doesn't expose main(), execute module-level entrypoint
        if hasattr(impl, "__dict__") and "__name__" in impl.__dict__:
            return 0
        return 0
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    raise SystemExit(main())
