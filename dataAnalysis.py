import pandas as pd
from pyecharts.charts import Bar, Geo, Map
# 导入输出图片工具
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot

from pyecharts import options as opts
from pyecharts.globals import GeoType

data = pd.read_csv('data\\各地区不同文物等级文物数量.csv').iloc[1:, :]
bar = Bar()
bar.add_xaxis(data['地区'].tolist())
bar.add_yaxis('国家级',data['国家级'].tolist())
bar.add_yaxis('省级',data['省级'].tolist())
bar.add_yaxis('市级',data['市级'].tolist())
bar.set_global_opts(title_opts=opts.TitleOpts(title="各地区不同文物等级文物数量",subtitle='A和B公司'),
                   toolbox_opts=opts.ToolboxOpts(is_show=True))
bar.set_series_opts(label_opts=opts.LabelOpts(position="top"))
bar.render("各地区不同文物等级文物数量.html")


