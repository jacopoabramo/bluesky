"""
This module creates an instance of the RunEngine and configures it. None of
this is *essential* but it is extremely useful and generally recommended.
"""
import os
import logging
import history
from .run_engine import RunEngine
from .legacy_scans import LegacyAscan, LegacyDscan, LegacyCount
from .register_mds import register_mds


logger = logging.getLogger(__name__)


### Set up a History object to handle peristence (scan ID, etc.)

SEARCH_PATH = []
ENV_VAR = 'BLUESKY_HISTORY_LOCATION'
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend([os.path.expanduser('~/.config/bluesky/bluesky_history.db'),
                    '/etc/bluesky/bluesky_history.db'])


def get_history():
    # Plan A: Find an existing db.
    for path in SEARCH_PATH:
        if os.path.isfile(path):
            try:
                h = history.History(path)
            except IOError:
                continue
            else:
                logger.debug('Using History file at %s' % path)
                return h
    # Plan B: Make new db.
    for path in SEARCH_PATH:
        if os.path.isdir(os.path.dirname(path)):
            try:
                h = history.History(path)
            except IOError:
                continue
            else:
                logger.debug('Created History file at %s' % path)
                return h
            return history.History(path[0])
    # Plan C: Use an in-memory sqlite db.
    logger.debug('Storing History in memory; it will not persist.')
    return history.History(':memory:')


RE = RunEngine(get_history())
register_mds(RE)  # subscribes to MDS-related callbacks
ascan = LegacyAscan(RE)
dscan = LegacyDscan(RE)
ct = LegacyCount(RE)