from importlib import resources

from allophant import package_data


RESOURCES = resources.files(package_data)

DEFAULT_CONFIG_FILE = "default_config.toml"
DEFAULT_CONFIG_PATH = str(RESOURCES / DEFAULT_CONFIG_FILE)
ALLOPHOIBLE_PATH = RESOURCES / "allophoible.csv"
DEFAULT_DIALECTS_PATH = RESOURCES / "default_dialects.json"
PHONEME_REPLACEMENTS_PATH = RESOURCES / "espeakng_phoneme_replacements.json"
