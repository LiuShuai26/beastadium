"""

python -m save_model --env=Walker --experiment=walker_0

"""
import sys

from sample_factory.save_model import save
from train import parse_custom_args, register_custom_components


def main():  # pragma: no cover
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args(evaluation=True)
    status = save(cfg, 74, 1024)
    return status




if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
