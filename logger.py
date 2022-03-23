import sys
import logging
import datetime


class Logger(object):
    def __init__(self, log_filename: str = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

        if log_filename is not None:
            log_filename = f"./logs/{log_filename}.log"
        else:
            log_filename = f"./logs/log-{get_current_time()}.log"

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 文件流
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # 终端
        stdf = logging.StreamHandler(sys.stdout)
        stdf.setLevel(logging.INFO)
        stdf.setFormatter(formatter)
        self.logger.addHandler(stdf)

    def info(self, info):
        self.logger.info(info)
        sys.stdout.flush()


def get_current_time():
    """get current time"""
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"


if __name__ == "__main__":
    L = Logger("12345")
    for i in range(10):
        L.info("Hello")
