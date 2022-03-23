import datetime

def get_current_time():
    '''current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}:{utc_plus_8_time.minute:0>2d}:{utc_plus_8_time.second:0>2d}"
    return f"{ymd} {hms}"

class Log:
    def __init__(self, log_filename=None) -> None:
        self._time = None
        self._msg = None
        self._level = None
        self._log_file = f"./logs/{self._get_current_time()}.log" if log_filename == None else log_filename

    def _get_current_time(self):
        utc_plus_8_time = datetime.datetime.now()
        ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
        hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
        return f"{ymd}_{hms}"
    
    def print_msg(self):
        msg = f"{self._time} | {self._level}: {self._msg}"
        print(msg)
        with open(self._log_file, mode="a") as f:
            f.write(msg+"\n")
            f.flush()

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
    L = log()
    for i in range(10):
        L.warning("-"*42+"ok")
        L.info("-"*42+"ok")
    