
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


data1 = pd.read_excel('data/教育资源/公办本科教育经费.xlsx')
data2 = pd.read_excel('data/教育资源/武汉市本科高校教育经费、师资力量.xlsx')
data3 = pd.read_excel('data/教育资源/武汉市高中分布数量.xlsx')
data4 = pd.read_excel('data/教育资源/高中大学分布数量.xlsx')

data1 = data1.rename(columns={'所在地': '地区', '2022（亿元）': '2022', '2023（亿元）': '2023'})
data2 = data2.rename(columns={'所在地': '地区', '2022（亿元）': '2022', '2023（亿元）': '2023'})
data3 = data3.dropna(how='any')
data3 = data3.replace('东湖生态旅游风景区文化教育局', '东湖风景区')
data3 = data3.replace('武汉东湖新技术开发区教育局代管', '东湖开发区')
data3 = data3.replace('武汉经济技术开发区教育局代管', '武汉开发区')
data4 = data4.rename(columns={'所在区': '地区'})
data4 = data4.iloc[:-1, :]
# print(data4['地区'])


# 地区公办\民办高校经费
df_budget2022_Private = data2.groupby(['办学性质', '地区'])['2022'].sum().to_frame().reset_index()
df_budget2023_Private = data2.groupby(['办学性质', '地区'])['2023'].sum().to_frame().reset_index()

# 地区高中学校、教师、学生数量
df_budget2022_teacher = data3.groupby('地区')['老师数量'].sum().to_frame().reset_index()
df_budget2023_student = data3.groupby('地区')['学生人数'].sum().to_frame().reset_index()

df_concat = pd.merge(data4, df_budget2022_Private, on='地区', how='left')
df_concat = pd.merge(df_concat, df_budget2023_Private[['地区', '2023']], on='地区', how='left')
df_concat = pd.merge(df_concat, df_budget2022_teacher, on='地区', how='left')
df_concat = pd.merge(df_concat, df_budget2023_student, on='地区', how='left')

df_concat.to_csv('data\\教育资源分析表.csv', index=False)
# print(df_concat)

# -------------------------------------------------------------------------------------------------------------------
pos_a = ['江岸区', '江汉区', '硚口区', '汉阳区', '武昌区', '青山区', '洪山区', '武汉开发区', '东湖开发区', '东湖风景区', '东西湖区', '汉南区', '蔡甸区', '江夏区', '黄陂区', '新洲区']
pos_b = ['江岸区', '江汉区', '硚口区', '汉阳区', '武昌区', '青山区', '洪山区', '武汉开发区', '技术开发区', '风景区', '东西湖区', '汉南区', '蔡甸区', '江夏区', '黄陂区', '新洲区']
data5 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各区博物馆+其藏品数量+其他信息/博物馆名录一览表_2022-12-29.xlsx')
data6 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各区博物馆+其藏品数量+其他信息/武汉市各区博物馆+藏品数量+其他信息.xlsx')
data7 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各区博物馆+其藏品数量+其他信息/武汉市文物保护资质单位名录.xlsx')
data8 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各地区国家级+省级+市级文物保护单位名录数据集/武汉市各地区全国重点文物保护单位名录_2021-05-24.xlsx')
data9 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各地区国家级+省级+市级文物保护单位名录数据集/武汉市各地区市级文物保护单位名录_2021-05-25.xlsx')
data10 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各地区国家级+省级+市级文物保护单位名录数据集/武汉市各地区省级文物保护单位名录_2021-05-24.xlsx')
data11 = pd.read_excel('data/文化遗产数据集以及相关数据说明/武汉市各地区国家级+省级+市级文物保护单位名录数据集/武汉市各地区重点文物保护单位分布表.xlsx')
data12 = pd.read_excel('data/补充数据/武汉市2019年教育经费情况.xls', header=None)
data13 = pd.read_excel('data/补充数据/武汉市各地区博物馆参观人数+教育活动举办次数xlsx.xlsx')

def getPos(x):
    for p in pos_b:
        if p in x:
            return p

    if '武昌' in x:
        return '武昌区'
    return '其他'


def getName(x):
    if '武汉市' == x[:3]:
        return x[3:]
    elif '武汉' == x[:2]:
        return x[2:]
    else:
        return x


def getMuseum(x):
    for m in museumList:
        if m in x:
            return m

    return x


data6 = data6.iloc[1:, :]
data6['地区'] = data13['博物馆分布区域'].apply(lambda x: getPos(x))
data6['博物馆名字'] = data6['博物馆名称'].apply(lambda x: getName(x))
museumList = data6['博物馆名字'].values.tolist()
data6 = data6.iloc[:, 3:]

data5 = data5.sort_values('博物馆名字').drop_duplicates()
data5['地区'] = data5['地址'].apply(lambda x: getPos(x))
data5['博物馆名字'] = data5['博物馆名字'].apply(lambda x: getMuseum(x))
data5 = data5.rename(columns={'性质': '博物馆性质'})
data5 = data5[['地区', '博物馆名字', '博物馆性质']]


data8['地区'] = data8['文物保护单位地址'].apply(lambda x: getPos(x))
data8 = data8.rename(columns={'文物等级': '国家级'})
data8 = data8.groupby('地区')['国家级'].count().to_frame().reset_index()

data9['地区'] = data9['文物保护单位地址'].apply(lambda x: getPos(x))
data9 = data9.rename(columns={'文物等级': '市级'})
data9 = data9.groupby('地区')['市级'].count().to_frame().reset_index()

data10['地区'] = data10['文物保护单位地址'].apply(lambda x: getPos(x))
data10 = data10.rename(columns={'文物等级': '省级'})
data10 = data10.groupby('地区')['省级'].count().to_frame().reset_index()

data11['地区'] = data11['保护单位地址'].apply(lambda x: getPos(x))


museum = data5.groupby(['地区'])['博物馆名字'].count().to_frame().reset_index()
print(museum)
museum.to_csv('data\\各地区博物馆数量分布.csv', index=False)
relics = data11.groupby(['地区'])['保护单位地址'].count().to_frame().reset_index()
print(relics)
relics.to_csv('data\\各地区保护单位分布.csv', index=False)

data56 = pd.merge(data5, data6, on='博物馆名字', how='left')
print(data56)
data56.to_csv('data\\各地区博物馆分布.csv', index=False)
data6.to_csv('data\\各地区藏品数量分布.csv', index=False)
data8910 = pd.merge(data8, data9, on='地区', how='left')
data8910 = pd.merge(data8910, data10, on='地区', how='left')
print(data8910)

data8910.to_csv('data\\各地区不同文物等级文物数量.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------
data12 = data12.iloc[5:, :]
data12.columns = ['地区', '一般公共预算教育经费', '一般公共预算教育经费占一般公共预算支出比例', '一般公共预算教育经费本年比上年增长']
print(data12)
data12.to_csv('data\\武汉市2019年教育经费情况.csv', index=False)

data13 = data13.iloc[1:, :]
data13['博物馆名字'] = data13['博物馆名称'].apply(lambda x: getName(x))
data13['地区'] = data13['博物馆分布区域'].apply(lambda x: getPos(x))
visit = data13.groupby('地区')['参观人数（2020年）'].sum().to_frame().reset_index()
educationalActivities = data13.groupby('地区')['教育活动（2020年）'].sum().to_frame().reset_index()
# print(data13)


educationalActivities_visit = pd.merge(visit, educationalActivities, how='left', on='地区')
print(educationalActivities_visit)
# print(educationalActivities)
educationalActivities_visit.to_csv('data\\参观人数教育活动.csv', index=False)


trainData = pd.merge(data6, educationalActivities_visit, how='left', on='地区')

trainData = pd.merge(trainData, df_budget2022_teacher, on='地区', how='left')
trainData = pd.merge(trainData, df_budget2023_student, on='地区', how='left')

df_budget2022_private = df_budget2022_Private[df_budget2022_Private['办学性质'] == '民办']
df_budget2022_public = df_budget2022_Private[df_budget2022_Private['办学性质'] == '公办']
df_budget2022_private.drop(columns=['办学性质'], inplace=True)
df_budget2022_public.drop(columns=['办学性质'], inplace=True)
print(df_budget2022_private)
print(df_budget2022_public)
private_cols = [x+'_民办' if x != '地区' else x for x in df_budget2022_private.columns]
public_cols = [x+'_公办' if x != '地区' else x for x in df_budget2022_public.columns]
df_budget2022_private.columns = private_cols
df_budget2022_public.columns = public_cols



trainData = pd.merge(trainData, df_budget2022_private, on='地区', how='left')
trainData = pd.merge(trainData, df_budget2022_public, on='地区', how='left')

df_budget2023_private = df_budget2023_Private[df_budget2023_Private['办学性质'] == '民办']
df_budget2023_public = df_budget2023_Private[df_budget2023_Private['办学性质'] == '公办']
df_budget2023_private.drop(columns=['办学性质'], inplace=True)
df_budget2023_public.drop(columns=['办学性质'], inplace=True)
private_cols = [x+'_民办' if x != '地区' else x for x in df_budget2023_private.columns]
public_cols = [x+'_公办' if x != '地区' else x for x in df_budget2023_public.columns]
df_budget2023_private.columns = private_cols
df_budget2023_public.columns = public_cols

trainData = pd.merge(trainData, df_budget2023_private, on='地区', how='left')
trainData = pd.merge(trainData, df_budget2023_public, on='地区', how='left')

trainData.to_csv('data\\dataTrain.csv', index=False)
print(trainData)
