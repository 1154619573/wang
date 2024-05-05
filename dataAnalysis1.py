import pandas as pd
from pyecharts.charts import Bar, Geo, Map
# 导入输出图片工具
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot

from pyecharts import options as opts
from pyecharts.globals import GeoType


data1 = pd.read_excel('data/教育资源/高中大学分布数量.xlsx')
data1 = data1.iloc[:-1, :]
data1.fillna(0, inplace=True)

quxian = data1['所在区'].values
print(quxian)
values3 = data1['大学数量'].values

# 获取符合pyecharts的数据格式
data_num = [list(z) for z in zip(quxian, values3)]

# 获取数据中的最大值
max_num = int(values3.max())

map3 = (
    Map()
        .add("武汉大学分布", data_num, "武汉")
        .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-VisualMap（连续型）"),
        visualmap_opts=opts.VisualMapOpts(max_=max_num)

    )
)
map3.render("武汉大学分布.html")


