
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


data = pd.read_csv('data/教育资源分析表.csv')
print(data.shape)
data = data.drop_duplicates('地区')
print(data)

num = len(data)
xticks = data['地区'].values.tolist()
plt.plot(range(num), data['老师数量']/1000, 'r', label='老师数量(千人)')
plt.plot(range(num), data['学生人数']/1000, label='学生人数(千人)')
plt.bar(range(num), data['高中数量'], label='高中数量')
plt.bar(range(num), data['大学数量'], label='大学数量')
plt.ylabel('数量')
plt.xlabel('地区')
plt.title('教育资源分布')
plt.xticks([i for i in range(len(xticks))], xticks, color='blue',rotation=60)
plt.legend()
# plt.savefig('教育资源分布.png')
# plt.close()
plt.show()


school = data['办学性质'].value_counts()

data = school.values
labels = school.index
plt.figure(figsize = (10,10))  # 设置画布大小
patches,l_text,p_text = plt.pie(data,labels = labels,autopct = '%1.2f%%',
        pctdistance = 0.4,labeldistance = 0.6)
plt.title("公办民办学校比例", fontsize=40)  # 设置标题
for t in l_text:
    t.set_size(30)
for t in p_text:
    t.set_size(30)
plt.axis('equal')
# plt.savefig('公办民办学校比例.png')
# plt.close()
plt.show()




data2 = pd.read_csv('data/各地区博物馆分布.csv')

parts = data2['博物馆性质'].value_counts()

data = parts.values
labels = parts.index
plt.figure(figsize = (10,10))  # 设置画布大小
patches,l_text,p_text = plt.pie(data,labels = labels,autopct = '%1.2f%%',
        pctdistance = 0.4,labeldistance = 0.6)
plt.title("不同性质博物馆比例", fontsize=40)  # 设置标题
for t in l_text:
    t.set_size(30)
for t in p_text:
    t.set_size(30)
plt.axis('equal')
# plt.savefig('不同性质博物馆比例.png')
# plt.close()
plt.show()


data3 = pd.read_csv('data/各地区博物馆数量分布.csv')
data4 = pd.read_csv('data/武汉市2019年教育经费情况.csv')
data4['地区'] = data4['地区'].apply(lambda x: x.strip())
data4 = data4.iloc[1:, :]
data34 = pd.merge(data4, data3, how='left', on='地区')

num = len(data34)
xticks = data34['地区'].values.tolist()
plt.plot(range(num), data34['博物馆名字'], '-o', c='r', label='博物馆数量')
plt.bar(range(num), data34['一般公共预算教育经费'], label='教育经费')
plt.bar(range(num), data34['一般公共预算教育经费占一般公共预算支出比例'], label='教育经费占比')
plt.bar(range(num), data34['一般公共预算教育经费本年比上年增长'], label='年增长')

plt.ylabel('数量')
plt.xlabel('地区')
plt.title('教育资源分布')
plt.xticks([i for i in range(len(xticks))], xticks, color='blue',rotation=60)
plt.legend()
# plt.savefig('教育资源和博物馆分布.png')
# plt.close()
plt.show()


data5 = pd.read_csv('data\\dataTrain.csv')
print(data5)

x = data5['教育活动（2020年）']
y = data5['参观人数（2020年）']
z = data5['学生人数']//1000

ax = plt.subplot(projection='3d')
ax.set_title('2020年博物馆教育活动与参观人数和学生人数的关系')
ax.scatter(x, y, z, c='r')

ax.set_xlabel('教育活动')
ax.set_ylabel('参观人数')
ax.set_zlabel('学生人数(千人)')

# plt.savefig('2020年博物馆教育活动与参观人数和学生人数的关系.png')
# plt.close()
plt.show()



x = data5['教育活动']
y = data5['参观人数']
z = data5['学生人数']//1000

ax = plt.subplot(projection='3d')
ax.set_title('博物馆教育活动与参观人数和学生人数的关系')
ax.scatter(x, y, z, c='r')

ax.set_xlabel('教育活动')
ax.set_ylabel('参观人数')
ax.set_zlabel('学生人数(千人)')

# plt.savefig('博物馆教育活动与参观人数和学生人数的关系.png')
# plt.close()
plt.show()



cols = data5.columns.tolist()
cols.remove('地区')
print(data5.info())
print(cols)

df = None
df_g = data5.groupby('地区')
for c in cols:
    if str(data5[c].dtype) == 'object':
        df_t = df_g[c].count().to_frame().reset_index()
    else:
        df_t = df_g[c].sum().to_frame().reset_index()

    if df is None:
        df = df_t
    else:
        df = pd.merge(df, df_t, on='地区', how='left')

print(df)
df = df.iloc[:, 1:]
df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
import seaborn as sns
#引入seaborn库
plt.figure(1)
sns.heatmap(df)#绘制new_df的矩阵热力图
#plt.savefig#('特征相关性热力图.png')
# plt.close()
plt.show()
