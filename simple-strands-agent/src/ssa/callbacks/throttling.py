import logging
import time

LOG = logging.getLogger(__file__)

MAX_RETRIES = 4
RETRY_DELAY = 60

class ThrottlingCallback:
    def __init__(self, retry_delay: int = RETRY_DELAY, max_attempts: int = MAX_RETRIES, **kwargs):
        self.retry_delay = retry_delay
        self.max_attempts = max_attempts

    def __call__(self, *args, **kwargs):
        event_delay = kwargs.get("event_loop_throttled_delay")
        if event_delay is not None and event_delay//2 < self.retry_delay:
            # delay of event_delay/2 is happened already
            LOG.info(f"Throttling exception: Additional delay by {self.retry_delay - event_delay//2} sec")
            time.sleep(self.retry_delay - event_delay//2)