import datetime

def get_current_time():
    '''get current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"


def print_time(time_elapsed, epoch=False):
    """打印程序执行时长"""
    time_hour = time_elapsed // 3600
    time_minite = (time_elapsed % 3600) // 60
    time_second = time_elapsed % 60
    if epoch:
        print(f"\nCurrent epoch take time: {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")
    else:
        print(f"\nAll complete in {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")