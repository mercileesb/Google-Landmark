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

import csv


def download_similar_image(image_path):
    org_image_path = image_path

    print(org_image_path)

    idx = org_image_path.split('/')[1]
    filename = org_image_path.split('/')[2].split('.')[0]

    # dst_image_folder = os.path.join(download_dst_folder,  org_image_path.split('/')[1])
    return_data = list()

    driver = load_driver('chromedriver')
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

    ActualImages = [] # contains the link for Large original images, type of  image

    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))
    for i, (img , Type) in enumerate(ActualImages):
        try:
            return_data.append(['{}_{}'.format(filename, i), img, idx])
            # req = Request(img)  #, headers={'User-Agent': header})
            # urlo = urlopen(req)
            # raw_img = urlo.read()
            # if len(Type) == 0:
            #     f = open(dst_image_folder + '_' + str(i) + ".jpg", 'wb')
            # elif Type == 'jpg' or Type == 'png':
            #     f = open(dst_image_folder + '_' + str(i) + "." + Type, 'wb')
            # f.write(raw_img)
            # f.close()
        except:
            continue

    driver.close()
    return return_data


def main():
    org_dir = 'image'

    filelists = list()
    target_dir = os.path.normpath(org_dir)  # remove trailing separator.
    for (path, dir, files) in os.walk(target_dir):
        for fname in files:
            paths = path.replace('\\', '/')
            folder_name = paths.split('/')[-1]
            org_filename = str(paths + "/" + fname)
            filelists.append(org_filename)

    pool = multiprocessing.Pool(processes=4)  # Num of CPUs
    return_value = pool.map(download_similar_image, filelists)
    pool.close()
    pool.terminate()

    with open('download.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['id', 'url', 'landmark_id'])

        for r_v in return_value:
            for item in r_v:
                csvwriter.writerow(item)


if __name__ == '__main__':
    main()
