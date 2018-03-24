from __future__ import print_function

import os, multiprocessing

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from driver.utils import *
from bs4 import BeautifulSoup

import json

try:
    from urllib.request import Request, urlopen
except:
    from urllib2 import Request, urlopen



def download_similar_image(image_path):
    org_image_path = image_path[0]
    dst_image_folder = image_path[1]

    print(org_image_path)

    # dst_image_folder = os.path.join(download_dst_folder,  org_image_path.split('/')[1])

    driver = load_driver("chromedriver")
    driver.get("https://www.google.com/imghp?h")

    image_path = os.path.join(os.getcwd(), org_image_path).replace('\\', '/')

    driver.find_element_by_xpath('//*[@id="gs_st0"]/a[1]').click()
    driver.find_element_by_xpath('//*[@id="qbug"]/div/a').click()
    file_input = driver.find_element_by_xpath('//*[@id="qbfile"]')
    file_input.send_keys(image_path)
    driver.find_element_by_xpath('//*[@id="imagebox_bigimages"]/g-section-with-header/div[1]/h3/a').click()

    y_waitkey = 50
    y_before = [-1 for x in range(y_waitkey)]

    while True:
        ActionChains(driver).send_keys(Keys.END).perform()
        smb_button = driver.find_element_by_xpath('//*[@id="smb"]')

        try:
            smb_button.click()

        except:
            y_now = driver.find_element_by_xpath('//*[@id="smc"]').rect['y']
            if y_before.count(y_now) >= y_waitkey:
                break
            y_before = y_before[1:]
            y_before.append(y_now)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.close()

    ActualImages = [] # contains the link for Large original images, type of  image

    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))
    for i, (img , Type) in enumerate(ActualImages):
        try:
            req = Request(img)  #, headers={'User-Agent': header})
            urlo = urlopen(req)
            raw_img = urlo.read()
            if len(Type) == 0:
                f = open(dst_image_folder + '_' + str(i) + ".jpg", 'wb')
            else:
                if Type == 'jpg' or Type == 'png':
                    f = open(dst_image_folder + '_' + str(i) + "." + Type, 'wb')
            f.write(raw_img)
            f.close()
        except:
            continue




def main():
    org_dir = 'image'
    dst_dir = 'donwload'

    filelists = list()
    try:
        os.mkdir(dst_dir)
    except:
        pass

    target_dir = os.path.normpath(org_dir)  # remove trailing separator.
    for (path, dir, files) in os.walk(target_dir):
        for fname in files:
            paths = path.replace('\\', '/')
            folder_name = paths.split('/')[-1]
            org_filename = str(paths + "/" + fname)

            dst_fname = fname.split('.')[0]

            dst_filename = os.path.join(dst_dir, folder_name, dst_fname).replace('\\', '/')
            fullfname = [org_filename, dst_filename]
            filelists.append(fullfname)

            try:
                dst_path = os.path.join(dst_dir, folder_name)
                os.mkdir(dst_path)
            except:
                continue


    # for i in filelists:
    #     download_similar_image(i)

    pool = multiprocessing.Pool(processes=4)  # Num of CPUs
    pool.map(download_similar_image, filelists)
    pool.close()
    pool.terminate()


if __name__ == '__main__':
    main()
