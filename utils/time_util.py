import datetime

def get_current_time():
    '''get current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"