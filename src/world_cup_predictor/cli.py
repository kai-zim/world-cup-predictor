import logging

from world_cup_predictor.config import Settings
from world_cup_predictor.logging_config import configure_logging


def main() -> None:
    configure_logging()
    settings = Settings()
    logging.getLogger(__name__).info("Loaded configuration with n_simulations=%s", settings.n_simulations)


if __name__ == "__main__":
    main()
