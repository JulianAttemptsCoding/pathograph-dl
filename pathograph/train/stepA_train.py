from __future__ import annotations

def main(argv=None):
    # Delegate to the current training entrypoint
    from pathograph.train.train_stmm import main as _main
    return _main(argv)

if __name__ == '__main__':
    raise SystemExit(main())
