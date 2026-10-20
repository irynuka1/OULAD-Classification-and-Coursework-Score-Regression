"""Command line entry point: ``python -m oauld <eda|classify|regress|all> [--mode dev|submit]``."""
import argparse

from . import classification, eda, regression


def main() -> None:
    parser = argparse.ArgumentParser(prog="oauld", description=__doc__)
    parser.add_argument("task", choices=["eda", "classify", "regress", "all"])
    parser.add_argument("--mode", choices=["dev", "submit"], default="dev",
                        help="dev: fit on train, score on the labelled hold-out. "
                             "submit: fit on train + hold-out, predict the private test set.")
    parser.add_argument("--gpu", action="store_true", help="train CatBoost on GPU 0")
    args = parser.parse_args()

    if args.task in ("eda", "all") and args.mode == "dev":
        eda.run()
    if args.task in ("classify", "all"):
        classification.run(args.mode, args.gpu)
    if args.task in ("regress", "all"):
        regression.run(args.mode, args.gpu)


if __name__ == "__main__":
    main()
