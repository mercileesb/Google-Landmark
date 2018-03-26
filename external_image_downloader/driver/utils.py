from selenium import webdriver
import platform
import struct

driver_root = None


def get_driver_root(global_save=True):
    os_ver = platform.system()
    driver_folder = 'driver/'

    if os_ver in 'Windows':
        driver_folder = driver_folder + 'windows/'

    # elif os_ver in 'Darwin':
    #     driver_root = driver_root + 'mac/'

    elif os_ver in {'Linux', 'Linux2'}:
        os_bit = struct.calcsize("P") * 8
        if os_bit == 32:  # bit
            driver_folder = driver_folder + 'linux32/'
        elif os_bit == 64:  # bit
            driver_folder = driver_folder + 'linux64/'

    else:
        print('Not supported OS')
        exit()

    if global_save:
        global driver_root

    driver_root = driver_folder
    return driver_root


def load_driver(driver_type='phantomjs', driver_folder=driver_root):
    if driver_folder is None:
        get_driver_root()
        global driver_root
        driver_folder = driver_root

    driver = None

    if driver_type == 'phantomjs':
        driver = webdriver.PhantomJS(driver_folder + 'phantomjs')

    elif driver_type == 'chromedriver':
        driver = webdriver.Chrome(driver_folder + 'chromedriver')

    else:
        print("Not supported types.")
        exit()

    return driver
