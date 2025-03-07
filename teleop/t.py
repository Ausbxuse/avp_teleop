import time


def sleep_until_mod33(time_curr):
    integer_part = int(time_curr)
    decimal_part = time_curr - integer_part
    ms_part = int(decimal_part * 1000) % 1000

    next_ms_part = ((ms_part // 33) + 1) * 33 % 1000

    next_capture_time = integer_part + next_ms_part / 1000
    print(next_capture_time - time_curr)


sleep_until_mod33(time.time())
