from lxml import etree
import requests
import random
import time
import csv
import pandas as pd
import numpy as np
import re
# 设置请求头参数：User-Agent, cookie, referer

headers = {
    # 随机生成User-Agent
    'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.1311 SLBChan/105", #"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    # 不同用户不同时间访问，cookie都不一样，根据自己网页的来，获取方法见Python之反爬虫手段
    'cookie' : '_T_WM=54639083573; WEIBOCN_FROM=1110006030; MLOGIN=0; XSRF-TOKEN=d0b7c1; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D100103type%253D1%2526q%253D%25E6%259C%25BA%25E7%25A5%25A8%26fid%3D100103type%253D1%2526q%253D%25E6%259C%25BA%25E7%25A5%25A8%26uicode%3D10000011',
    # 设置从何处跳转过来(本次可以不用设置)
    #'referer': '',
}
columns = ['网名', '国家', '省会', '城市', '手机终端', '文本']
df = pd.DataFrame(columns=columns)
for num in range(100):  # 由于时间原因，只爬取100页，约1000条左右数据
    # 首页网址URL
    url = 'https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D61%26q%3D%E6%9C%BA%E7%A5%A8%26t%3D&page_type=searchall&page=' + str(
        num + 1)
    # print(url)

    # time.sleep(1)
    # 请求发送
    response = requests.get(url=url, headers=headers).json()
    card_list = response.get('data').get('cards')
    print("******************开始爬取第", str(num + 1), "页******************")
    for i in range(len(card_list)):
        values = []
        if 'card_group' in card_list[i] and 'mblog' in card_list[i]:
            dict1 = card_list[i].get('card_group')[0].get('mblog')
            values.append(dict1.get('user').get('screen_name'))  # 网名
            values.append(dict1.get('status_country'))  # 国家
            values.append(dict1.get('status_province'))  # 省会
            values.append(dict1.get('status_city'))  # 城市
            values.append(dict1.get('source'))  # 手机终端
            values.append(re.sub("[A-Za-z0-9\!\%\[\]\,\。\<\=\"\:\/\.\/\?\&\-\>]", "", dict1.get('text')))  # 文本
            df = df.append(pd.DataFrame([values], columns=df.columns))
            print(values)

        elif 'mblog' in card_list[i]:
            dict1 = card_list[i].get('mblog')
            values.append(dict1.get('user').get('screen_name'))  # 网名
            values.append(dict1.get('status_country'))  # 国家
            values.append(dict1.get('status_province'))  # 省会
            values.append(dict1.get('status_city'))  # 城市
            values.append(dict1.get('source'))  # 手机终端
            values.append(
                re.sub("[A-Za-z0-9\!\%\[\]\,\。\<\=\"\:\/\.\/\?\&\-\>\_\; \_\ ; \']", "", dict1.get('text')))  # 文本
            df = df.append(pd.DataFrame([values], columns=df.columns))
            print(values)
    print("******************第", str(num + 1), "页爬取成功******************")
df.head()  # 展示前5行
df.to_csv('微博文化遗产和教育资源相关文章.csv')  #这里根据需要命名自己爬取帖子的文件名字