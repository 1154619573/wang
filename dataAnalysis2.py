import pandas as pd
import numpy as np
from pyecharts.charts import Bar, Geo, Map
from pyecharts import options as opts

data1 = pd.read_excel('data/教育资源/高中大学分布数量.xlsx')
data1.columns = ['地区', '高中数量', '大学数量']
data2 = pd.read_csv('data\\各地区博物馆数量分布.csv')
data1 = data1.iloc[2:, :]

data1 = pd.merge(data1, data2, on='地区', how='left')
data1.fillna(0, inplace=True)

quxian = data1['地区'].values

values3 = data1['博物馆名字'].values
print(values3)


# 获取符合pyecharts的数据格式
data_num = [list(z) for z in zip(quxian, values3)]

# 获取数据中的最大值
max_num = int(values3.max())

map3 = (
    Map()
        .add("各地区博物馆数量分布", data_num, "武汉")
        .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-VisualMap（连续型）"),
        visualmap_opts=opts.VisualMapOpts(max_=max_num)

    )
)
map3.render("各地区博物馆数量分布.html")


