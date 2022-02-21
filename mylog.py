import datetime

def get_current_time():
    '''UTC+8 time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}:{utc_plus_8_time.minute:0>2d}:{utc_plus_8_time.second:0>2d}"
    return f"{ymd} {hms}"

class Log:
    def __init__(self) -> None:
        self._time = None
        self._msg = None
        self._level = None
    
    def print_msg(self):
        print(f"{self._time} | {self._level}: {self._msg}")

    def info(self, msg: str):
        self._time = get_current_time()
        self._level = "INFO"
        self._msg = msg
        self.print_msg()
    
    def error(self, msg: str):
        self._time = get_current_time()
        self._level = "ERROR"
        self._msg = msg
        self.print_msg()
    
    def debug(self, msg: str):
        self._time = get_current_time()
        self._level = "DEBUG"
        self._msg = msg
        self.print_msg()
    
    def warning(self, msg: str):
        self._time = get_current_time()
        self._level = "WARNING"
        self._msg = msg
        self.print_msg()

def log():
    return Log()

if __name__ == '__main__':
    L =log()
    L.warning("Hello")
    