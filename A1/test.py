import platform

if platform.system().lower() == 'windows':
    print("windows!")
else:
    print("mac!")