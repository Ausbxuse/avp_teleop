import time


def sleep_until_mod33(time_curr):
    integer_part = int(time_curr)
    decimal_part = time_curr - integer_part
    ms_part = int(decimal_part * 1000) % 100

    print(time_curr)
    next_ms_part = ((ms_part // 33) + 1) * 33 % 100
    hundred_ms_part = int(decimal_part * 10 % 10)
    if next_ms_part == 32:
        hundred_ms_part += 1
    print(next_ms_part, hundred_ms_part)

    next_capture_time = integer_part + next_ms_part / 1000 + hundred_ms_part / 10
    if (next_capture_time - time_curr) < 0:
        next_capture_time += 1
    print(next_capture_time)


sleep_until_mod33(0.999)
