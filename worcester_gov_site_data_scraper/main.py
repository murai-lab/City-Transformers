# import docutils.nodes
import requests
from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
import pandas as pd
import numpy as np
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import time
import urllib3
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

pd.set_option('display.max_columns', None)

# df = pd.DataFrame()
# df['PID'] = ''
# df['url'] = ''
# df['construction_date'] = ''
# df['address'] = ''
# df['style'] = ''

#print(df)

data_dict = {
    'PID':[],
    'URL':[],
    'construction_date':[],
    'address':[],
    'style':[],
    'image':[],
    'location':[],
    'acctnum':[]
}

def data_collector(pid_list):
    base_url = "https://gis.vgsi.com/worcesterma/Parcel.aspx?pid="
    for i in pid_list:
        try:
            pid = i
            url = base_url + str(i)

            # page = urlopen(url)
            page = requests.get(url, verify=False)

            soup = BeautifulSoup(page.content, features = 'html.parser')


            year_built = soup.find(id = 'MainContent_ctl01_lblYearBuilt')
            address = soup.find(id = 'MainContent_lblAddr1')
            location = soup.find(id = 'MainContent_lblLocation')
            acctnum = soup.find(id = 'MainContent_lblAcctNum')
            image = soup.find(id='MainContent_ctl01_imgPhotoLink')
            if image:
                image = str(image['href'])
            style = soup.find(id='MainContent_ctl01_grdCns')
            if style:
                style = style.find(class_='RowStyle')



            def remove_tags(data):
                if data is None:
                    return None

                for i in data(['style', 'script']):
                    i.decompose()

                return ' '.join(data.stripped_strings)

            if style:
                style = remove_tags(style)
                style = style[6:]
            #image = image.replace('\\','/')
            year_built = remove_tags(year_built)
            address = remove_tags(address)
            location = remove_tags(location)
            acctnum = remove_tags(acctnum)
            data_dict['PID'].append(i)
            data_dict['URL'].append(url)
            data_dict['construction_date'].append(year_built)
            data_dict['address'].append(address)
            data_dict['location'].append(location)
            data_dict['acctnum'].append(acctnum)
            data_dict['image'].append(image)
            data_dict['style'].append(style)
            #print(image)
            print(pid)
            # time.sleep(1)
        except Exception as e:
            print('pass' +str(pid))
            print(e)
            # time.sleep(1)

            pass

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.transpose()
    #print(df)
    # if file already exists, load, append and save
    try:
        df_old = pd.read_csv('test.csv')
        df = pd.concat([df_old, df])
    except:
        pass
    df.to_csv('test.csv', index=False)



pid_list = pd.read_csv('worcester_city_data.csv', usecols=['PID'])['PID'].sort_values().tolist()
data_collector(pid_list)


# year built
# path id=MainContent_ctl01_lblYearBuilt\"

# address
# path "#MainContent_lblAddr1"

# style table
# MainContent_ctl01_grdCns"

#photo
# MainContent_ct101_imgPhotoLink







#"MainContent_ctl01_imgPhotoLink"


#
#
# url = "https://gis.vgsi.com/worcesterma/Parcel.aspx?pid=3"
#
#
# page = urlopen(url)
# soup = BeautifulSoup(page, features = 'lxml')




# def remove_tags(data):
#     for i in data(['style', 'script']):
#         i.decompose()
#
#     return ' '.join(data.stripped_strings)

# image = soup.find(id = 'MainContent_ctl01_imgPhotoLink')
# image = str(image['href'])
#
# seg_image = image
# print(len(image))
# print(image)
#
#
# def url_image_convertor(image):
#     seg_image = image[:-26]
#     adj_image_1 = image[-25:-23]
#     adj_image_2 = image[-22:-20]
#     adj_image_3 = image[-19:-17]
#     adj_image_4 = image[-16:]
#     slash = '/'
#     new_image = slash + adj_image_1 + slash + adj_image_2 + slash + adj_image_3 + slash + adj_image_4
#
#     image_url = seg_image + new_image
#     return image_url
#

#print(url_image_convertor(image))

# img = mpimg.imread('.jpg')
# imgplot = plt.imshow(img)
# plt.show()

# for i in image:
#     if i == '\\':
#         print(True)
#     else:
#         print(False)
#
# image = image.replace('\\','/')
# print(image)
