class ScraperError(Exception):
    """Base class for expected scraper failures."""


class ConfigurationError(ScraperError):
    pass


class SiteUnavailableError(ScraperError):
    pass


class SecurityChallengeError(ScraperError):
    pass


class EditionNotFoundError(ScraperError):
    pass


class KPTANotFoundError(ScraperError):
    pass


class PriceNotFoundError(ScraperError):
    pass


class AmbiguousPriceError(ScraperError):
    pass
