#!/usr/bin/env python
# coding: utf-8

# <a id = '5.5'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Flight Price Prediction</strong></p>
# 以上为Notebook中的标题部分，定义了一个锚点和标题样式，用于展示“Flight Price Prediction”标题。

# ![images.jpg](attachment:14dc9a04-741e-4fd0-9ab9-e4e47b34f004.jpg)
# 加载一张图片文件作为展示，不属于代码逻辑的一部分。

# In[1]:


#importing libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')
#中文注释：
# 导入所需的Python库，包括数据处理(pandas、numpy)、可视化(seaborn、matplotlib)和机器学习相关(sklearn)的工具包。
# 通过warnings.filterwarnings('ignore')忽略一些不必要的警告信息。


# # Importing dataset
# 1.Since data is in form of excel file we have to use pandas read_excel to load the data.   
# 2.After loading it is important to check null values in a column or a row           
# 3.If it is present then following can be done,                                                                                   
# *       Filling NaN values with mean, median and mode using fillna() method                                                           
# *       If Less missing values, we can drop it as well 
#中文注释：
# 上方为对数据导入步骤的说明。


# In[2]:


pip install openpyxl
#中文注释：
# 安装openpyxl，以便pandas能够读取Excel文件。


# In[3]:


#importing data
df=pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')
df.head()
#中文注释：
# 使用pandas的read_excel函数读取Excel格式的数据集，并查看前几行数据。


# In[4]:


df.info()   #information about the data
#中文注释：
# 显示数据集df的概要信息，包括列名、数据类型、非空值数量。


# In[5]:


#describe about the data
df.describe()
#中文注释：
# 对数据集进行描述性统计，显示数值型特征的计数、均值、中位数、标准差、最小值、最大值等统计信息。


# In[6]:


df.shape
#中文注释：
# 显示数据集的行数和列数。


# In[7]:


#finding the null values
df.isnull().sum()
#中文注释：
# 统计各列的缺失值（空值）数量。


# In[8]:


import missingno as msno
msno.bar(df)
plt.show
#中文注释：
# 使用missingno库可视化缺失值分布情况，以柱状图显示每一列的有效数据条数。
# plt.show用于显示图形。


# ### **We observe 2 missing values,I can directly drop these,as these are very less.**
#中文注释：
# 观察到有两条缺失值，由于数量很少，可以直接删除这些行。


# In[9]:


#drop the nullvalues
df.dropna(inplace=True)
#中文注释：
# 删除数据中含有缺失值的行，并直接修改原数据（inplace=True）。


# In[10]:


df.isnull().sum()
#中文注释：
# 再次检查数据集中是否还有缺失值。


# <a id = '5.5'></a>
# <p style = "font-size : 25px; color : 'blue' ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Data Cleaning</strong></p>
# 中文注释：
# 添加标题，用于说明接下来进行数据清洗的步骤。


# In[11]:


df.dtypes # datatypes
#中文注释：
# 显示每个列的数据类型。


# #### The datatypes of Date_of_journey,Arrival_Time and Dep_Time is object.So,firstly we convert it into date and time for proper predicion.
#中文注释：
# Date_of_journey, Arrival_Time, Dep_Time为字符串类型，需要转换成日期时间格式，以便后续处理。


# In[12]:


def change_into_datetime(col):
    df[col]=pd.to_datetime(df[col])
#中文注释：
# 定义一个函数，将指定列转换为datetime格式。


# In[13]:


df.columns
#中文注释：
# 查看数据集中所有列名。


# In[14]:


for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)
#中文注释：
# 对日期和时间相关的列进行转换为日期时间类型。


# In[15]:


df.dtypes
#中文注释：
# 再次检查列的数据类型，确保转换成功。


# ##### Now ,we extract day and month from Date_of_journey and stored in 2 other columns.
# ##### Then there will be no use of 'Date_of_Journey' column,so we drop it.
#中文注释：
# 从Date_of_Journey中提取出天和月，并创建新列journey_day和journey_month，然后删除原Date_of_Journey列。


# In[16]:


df['journey_day']=df['Date_of_Journey'].dt.day
df['journey_month']=df['Date_of_Journey'].dt.month
#中文注释：
# 从Date_of_Journey中提取"日"和"月"信息并存入新列。


# In[17]:


df.head(10)
#中文注释：
# 查看前10行数据，验证新列已正确添加。


# In[18]:


df.drop('Date_of_Journey', axis=1, inplace=True)
#中文注释：
# 删除已无用的Date_of_Journey列。


# ##### From Arrival_time and Dept_time features,we extract hour and minutes and stored in new columns and drop these columns
#中文注释：
# 从Arrival_Time和Dep_Time中提取小时和分钟信息，以便后续分析。


# In[19]:


# function for extracting hour and minutes
def extract_hour(data,col):
    data[col+'_hour']=data[col].dt.hour
    
def extract_min(data,col):
    data[col+'_min']=data[col].dt.minute
    

def drop_col(data,col):
    data.drop(col,axis=1,inplace=True)
#中文注释：
# 定义三个函数，用于从时间列中提取小时、分钟，并删除原列。


# In[20]:


#call the function
# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time
extract_hour(df,'Dep_Time')
#中文注释：
# 从Dep_Time中提取小时信息并新建列Dep_Time_hour。

extract_min(df,'Dep_Time')
#中文注释：
# 从Dep_Time中提取分钟信息并新建列Dep_Time_min。

drop_col(df,'Dep_Time')
#中文注释：
# 删除原始的Dep_Time列。


# In[21]:


#extracting hour
extract_hour(df,'Arrival_Time')
#中文注释：
# 从Arrival_Time中提取小时信息。

#extracting min
extract_min(df,'Arrival_Time')
#中文注释：
# 从Arrival_Time中提取分钟信息。

#drop the column
drop_col(df,'Arrival_Time')
#中文注释：
# 删除原始的Arrival_Time列。


# In[22]:


df.head(10)
#中文注释：
# 查看数据，验证新的小时和分钟列已正确添加。


# ##### Lets Apply pre-processing on duration column,Separate Duration hours and minute from duration
#中文注释：
# 对Duration列进行处理，将持续时间分为小时和分钟。


# In[23]:


duration=list(df['Duration'])
for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]: # Check if duration contains only hour
             duration[i]=duration[i] + ' 0m' # Adds 0 minute
        else:
             duration[i]='0h '+ duration[i]
#中文注释：
# 将Duration列转换为标准形式，如果只有小时(h)，则补充0m；如果只有分钟(m)，则补充0h。
# 最终确保每个持续时间字符串格式为“xh ym”。


# In[24]:


df['Duration']=duration
#中文注释：
# 将标准化后的持续时间列表赋值回数据集中。


# In[25]:


df.head()
#中文注释：
# 查看数据，验证Duration列的格式已标准化。


# In[26]:


def hour(x):
    return x.split(' ')[0][0:-1]

def minutes(x):
    return x.split(' ')[1][0:-1]
#中文注释：
# 定义两个函数hour和minutes，用于从标准化的持续时间中提取小时和分钟部分的数值。


# In[27]:


df['dur_hour']=df['Duration'].apply(hour)
#中文注释：
# 对Duration列应用hour函数，提取小时值。


# In[28]:


df['dur_min']=df['Duration'].apply(minutes)
#中文注释：
# 对Duration列应用minutes函数，提取分钟值。


# In[29]:


df.head(10)
#中文注释：
# 查看处理后的数据，验证dur_hour和dur_min列已正确生成。


# In[30]:


drop_col(df,'Duration')
#中文注释：
# 删除原始的Duration列，因我们已提取了小时和分钟信息。


# In[31]:


df.dtypes
#中文注释：
# 检查数据类型，当前dur_hour和dur_min为字符串。


# In[32]:


df['dur_hour'] = df['dur_hour'].astype(int)
df['dur_min'] = df['dur_min'].astype(int)
#中文注释：
# 将dur_hour和dur_min转换为整数类型，以便后续分析或建模。


# In[33]:


df.dtypes
#中文注释：
# 再次检查数据类型。


# #### Finding the categorical value
#中文注释：
# 查找数据集中类型为object的列。


# In[34]:


column=[column for column in df.columns if df[column].dtype=='object']
column
#中文注释：
# 列出数据集中的对象类型列名称。


# #### Finding the cntinuous value
#中文注释：
# 查找数值（连续型）列。


# In[35]:


continuous_col =[column for column in df.columns if df[column].dtype!='object']
continuous_col
#中文注释：
# 列出数据集中非对象类型（数值型）的列名称。


# # Handling categorical data
#中文注释：
# 对分类变量进行编码处理。


# ### We are using two main Encoding Techniques to covert Categorical data into some numerical format
# #### Nominal data -- Data that are not in any order -->one hot encoding
# #### ordinal data -- Data are in order --> labelEncoder
#中文注释：
# 根据分类数据的属性，选择适当的编码方式。一致性数据用LabelEncoder，无序分类用One Hot Encoding。


# In[36]:


categorical = df[column]
categorical.head()
#中文注释：
# 将分类数据的列单独存为一个DataFrame(categorical)，方便后续处理。
# 显示前几行分类数据。


# In[38]:


categorical['Airline'].value_counts()
#中文注释：
# 查看Airline列中每个类别的出现频次。


# ## Airline vs Price Analysis
#中文注释：
# 对航空公司(Airline)与价格(Price)进行可视化分析。


# In[39]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Airline',y='Price',data=df.sort_values('Price',ascending=False))
#中文注释：
# 使用箱型图对不同航空公司的票价分布进行比较。


# ### From graph we can see that Jet Airways Business have the highest Price.
#中文注释：
# 根据图形可知Jet Airways Business票价最高，其余大部分航班价格中位数相近。


# #### Perform Total_Stops vs Price Analysis
#中文注释：
# 分析中途停靠次数与票价的关系。


# In[40]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Total_Stops',y='Price',data=df.sort_values('Price',ascending=False))
#中文注释：
# 使用箱型图对中途停靠次数(Total_Stops)与价格(Price)之间的关系进行可视化。


# In[41]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding
Airline=pd.get_dummies(categorical['Airline'],drop_first=True)
Airline.head()
#中文注释：
# 对Airline列执行独热编码，drop_first=True可避免虚拟变量陷阱(减少冗余列)。


# In[43]:


categorical['Source'].value_counts()
#中文注释：
# 查看出发地(Source)的各类别统计。


# In[44]:


#Source vs Price
plt.figure(figsize=(15,15))
sns.catplot(x='Source',y='Price',data=df.sort_values('Price',ascending=False),kind='boxen')
#中文注释：
# 使用boxen图查看不同出发地与价格分布的关系。


# In[45]:


#encoding of source column
source=pd.get_dummies(categorical['Source'],drop_first=True)
source.head()
#中文注释：
# 对Source列进行独热编码。


# In[46]:


categorical['Destination'].value_counts()
#中文注释：
# 查看目的地(Destination)列中各类别的分布。


# In[47]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Destination',y='Price',data=df.sort_values('Price',ascending=False))
#中文注释：
# 查看不同目的地的票价分布。


# In[48]:


#encoding of destination column
destination=pd.get_dummies(categorical['Destination'],drop_first=True)
destination.head()
#中文注释：
# 对Destination列进行独热编码。


# In[49]:


# now work on route column
categorical['Route'].value_counts()
#中文注释：
# 查看航线(Route)列中不同路线组合的频次。


# In[50]:


categorical['Route1']=categorical['Route'].str.split('→').str[0]
categorical['Route2']=categorical['Route'].str.split('→').str[1]
categorical['Route3']=categorical['Route'].str.split('→').str[2]
categorical['Route4']=categorical['Route'].str.split('→').str[3]
categorical['Route5']=categorical['Route'].str.split('→').str[4]
#中文注释：
# 将Route根据'→'分隔符拆分为多个子站点列(从Route1到Route5)。


# In[52]:


drop_col(categorical,'Route')
#中文注释：
# 删除原始Route列，因为我们已将其分解为多个列。


# In[53]:


categorical.isnull().sum()
#中文注释：
# 检查分裂后是否有空值。


# In[55]:


for i in ['Route3', 'Route4', 'Route5']:
    categorical[i].fillna('None',inplace=True)
#中文注释：
# 对Route3, Route4, Route5中可能存在的空值填入'None'。


# In[57]:


for i in categorical.columns:
    print('{} has total {} categories'.format(i,len(categorical[i].value_counts())))
#中文注释：
# 打印categorical中每列的类别数量。


# In[58]:


df.plot.hexbin(x='Arrival_Time_hour',y='Price',gridsize=15)
#中文注释：
# 使用hexbin图查看到达时间的小时与价格之间的关系分布。


# In[59]:


# Applying label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
#中文注释：
# 引入LabelEncoder对有序类别进行编码。


# In[60]:


for i in ['Route1', 'Route2', 'Route3', 'Route4', 'Route5']:
    categorical[i]=encoder.fit_transform(categorical[i])
#中文注释：
# 对分解出来的Route子列进行标签编码，将文字转换为数字表示。


# In[62]:


drop_col(categorical,'Additional_Info')
#中文注释：
# 删除Additional_Info列（如果不需要或其信息量不大）。


# In[63]:


categorical['Total_Stops'].unique()
#中文注释：
# 查看总停靠次数列的类别值。


# In[64]:


# encoding Total stops
dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)
#中文注释：
# 根据停靠次数映射为数字编码。


# In[66]:


drop_col(categorical,'Source')
drop_col(categorical,'Destination')
drop_col(categorical,'Airline')
#中文注释：
# 删除已用独热编码处理过的原始分类列（Source, Destination, Airline）。


# <a id = '5.5'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>After all preprocessing, Our data is ready for the modeling</strong></p> 
#中文注释：
# 标题：数据清洗和特征工程完成后，数据可用于建模。


# In[67]:


final_df=pd.concat([categorical,Airline,source,destination,df[continuous_col]],axis=1)
#中文注释：
# 将处理后的分类数据(categorical)、独热编码后的Airline、Source、Destination与连续特征列合并成一个最终数据集final_df。


# In[68]:


final_df.head()
#中文注释：
# 查看合并后的数据集前几行。


# In[69]:


pd.set_option('display.max_columns',33)
final_df.head()
#中文注释：
# 设置显示最大列数为33，方便查看完整数据。


# ## Check For Outliers 
#中文注释：
# 检查数据中的异常值(Outliers)。


# In[70]:


def plot(data,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(data[col],ax=ax1)
    sns.boxplot(data[col],ax=ax2)
#中文注释：
# 定义plot函数用于同时绘制特定列的数据分布图和箱型图，以识别异常值。


# In[71]:


plot(final_df,'Price')
#中文注释：
# 对价格列进行分布与箱型图检查，以识别票价中的异常值。


# ###  Handling outliers:
# #### As there is some outliers in price feature,so we replace it  with median.
#中文注释：
# 观察到Price列有异常值，将异常值替换为中位数，以降低异常值对模型的影响。


# In[72]:


final_df['Price']=np.where(final_df['Price']>=40000,final_df['Price'].median(),final_df['Price'])
#中文注释：
# 将所有大于等于40000的票价替换为中位数。


# In[73]:


plot(final_df,'Price')
#中文注释：
# 再次绘图，检查处理后价格分布情况。


# ### Seprate the dataset in X and Y columns
#中文注释：
# 将数据集分为特征(X)和目标变量(y)，其中y为价格Price。


# In[74]:


X=final_df.drop('Price',axis=1)
y=df['Price']
#中文注释：
# 从final_df中删除Price列获得特征X。
# 因为在final_df中做了处理，这里y直接从原始df中拿Price列，保持一致。


# # Feature Selection 
#中文注释：
# 使用特征选择方法来评估特征对目标的影响。


# In[75]:


from sklearn.feature_selection import mutual_info_classif
#中文注释：
# 导入互信息评分函数，用于估计特征与目标变量之间的相关性。


# In[76]:


mutual_info_classif(X,y)
#中文注释：
# 计算每个特征与目标之间的互信息值，用于评估特征重要性。


# In[77]:


imp = pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
imp
#中文注释：
# 将互信息值存入数据框，索引为特征名称。


# In[78]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)
#中文注释：
# 为互信息结果命名列并按重要性从高到低排序。


# # Models
#中文注释：
# 准备训练各种机器学习模型。


# In[79]:


# spiliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=123)
#中文注释：
# 将数据集分为训练集和测试集，测试集占比20%。


# In[80]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def predict(ml_model):
    print('Model is: {}'.format(ml_model))
    model= ml_model.fit(X_train,y_train)
    print("Training score: {}".format(model.score(X_train,y_train)))
    predictions = model.predict(X_test)
    print("Predictions are: {}".format(predictions))
    print('\n')
    r2score=r2_score(y_test,predictions) 
    print("r2 score is: {}".format(r2score))
    print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
    print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
    sns.distplot(y_test-predictions)  
#中文注释：
# 定义一个通用预测函数predict，用于训练给定的模型、评估性能并打印评价指标和误差分布。


# In[81]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
#中文注释：
# 导入回归模型：逻辑回归（用于分类，这里只是演示）、KNN回归、决策树回归、梯度提升回归以及随机森林回归。


# In[82]:


predict(RandomForestRegressor())
#中文注释：
# 使用随机森林模型进行训练和预测，并输出性能指标。


# In[83]:


predict(LogisticRegression())
#中文注释：
# 使用逻辑回归进行预测（虽然不太适合回归任务，此处仅作演示）。


# In[84]:


predict(KNeighborsRegressor())
#中文注释：
# 使用KNN回归模型进行预测。


# In[85]:


predict(DecisionTreeRegressor())
#中文注释：
# 使用决策树回归模型进行预测。


# In[86]:


from sklearn.svm import SVR
predict(SVR())
#中文注释：
# 使用支持向量回归(SVR)进行预测。


# In[87]:


predict(GradientBoostingRegressor())
#中文注释：
# 使用梯度提升回归模型进行预测。


# # Hypertunning the model
#中文注释：
# 对模型进行超参数搜索，以期提升模型性能。


# In[88]:


from sklearn.model_selection import RandomizedSearchCV
#中文注释：
# 使用随机搜索(RandomizedSearchCV)调优模型参数。


# In[89]:


random_grid = {
    'n_estimators' : [100, 120, 150, 180, 200,220],
    'max_features':['auto','sqrt'],
    'max_depth':[5,10,15,20],
    }
#中文注释：
# 定义随机搜索的参数网格，包括随机森林模型的树数量、特征选择方式和深度范围。


# In[90]:


rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1,)
rf_random.fit(X_train,y_train)
#中文注释：
# 使用随机搜索在给定参数范围内尝试最佳超参数组合，cv=3表示三折交叉验证，n_jobs=-1表示使用全部CPU核心加速。


# In[91]:


rf_random.best_params_
#中文注释：
# 输出随机搜索得到的最佳参数组合。


# In[92]:


prediction = rf_random.predict(X_test)
sns.displot(y_test-prediction)
#中文注释：
# 使用优化后的模型对测试集进行预测，并查看预测误差分布图。


# In[93]:


r2_score(y_test,prediction)
#中文注释：
# 计算最终模型的R²分数，查看性能改善情况。


# ## After hypertuning,the accuracy increases .
# 中文注释：
# 超参数调整后，模型的精度有所提升。


# #### IF YOU LIKE THE WORK,PLEASE UPVOTE IT.
#中文注释：
# 最后的说明和鼓励。该段不是核心代码逻辑的一部分。
