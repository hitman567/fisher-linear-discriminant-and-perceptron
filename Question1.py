
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import mpld3
#mpld3.enable_notebook()


# In[ ]:


data=pd.read_csv("dataset_1.csv")  #reading data
# data.head()


# In[3]:


# x=data['X']
# y=data['Y']
# plt.scatter(x,y,c=data["target"])


# In[4]:


data_groupby=data.groupby("target")  ##grouping data according to the target value
data_groupby.first()

# print(data_groupby)


# In[5]:


data0=data_groupby.get_group(0) # storing two classes in two different dataframes
data1=data_groupby.get_group(1)


# In[6]:


# data0.head()


# In[7]:


plt.scatter(data0['X'],data0["Y"],color="r")  ##plotting scatter graph using matplotlib
plt.scatter(data1['X'],data1["Y"],color="b")


# In[8]:


mean0=np.mean(data0)  ##finding means of both classes
mean1=np.mean(data1)


# print(mean0.shape)
overallmean=np.mean(data.drop(['target'],axis=1))  ##finding overall mean


# In[9]:


# mean0=mean0.as_matrix()
# mean1=mean1.as_matrix()
# overallmean=overallmean.as_matrix()

#print(mean0)
# In[10]:


mean0=np.resize(mean0,(2,1))  ##converting and storing mean as matrix
mean1=np.resize(mean1,(2,1))
overallmean=np.resize(overallmean,(2,1))

#print(mean0)
# In[11]:


# print(mean0.shape)
# print(mean1.shape)
# print(overallmean.shape)


# In[12]:


n0=data0["X"].count()   #counting number of rows in both dataset
n1=data1["X"].count()
# print(n1)
# print(n0)


# In[13]:


s0=np.matmul((mean0-overallmean)*n0,(mean0-overallmean).T) ## finding Sw
# print(s0)
s1=np.matmul((mean1-overallmean)*n1,(mean1-overallmean).T)
# print(s1)


# In[14]:


sb=s0+s1
sb=np.array(sb)  ##finding Sb
# print(sb.shape)
# print(sb)


# In[15]:


sw0=np.zeros(sb.shape)
for i in data0.itertuples():
    mat=np.zeros((2,1))
    mat[0][0]=i.X
    mat[1][0]=i.Y
    sw0=np.add(np.matmul((mat-mean0),(mat-mean0).T),sw0)

    


# In[16]:


# print(sw0)


# In[17]:


sw1=np.zeros(sb.shape)
for i in data1.itertuples():
    mat=np.zeros((2,1))
    mat[0][0]=i.X
    mat[1][0]=i.Y
    sw1=np.add(np.matmul((mat-mean1),(mat-mean1).T),sw1)

    


# In[18]:


# print(sw1)


# In[19]:


sw=sw0+sw1
# print(sw)


# In[20]:


swinv=np.linalg.inv(sw)  ##finding inverse of Sw
# print(swinv)


# In[21]:


direction=np.matmul(swinv,(mean1-mean0))  ## finding direction of linear separable line
# print(direction)
# print(direction.shape)


# In[22]:


proj_zero=[]
# direc=direction[0][0]**2+direction[1][0]**2
# print(direc)



# In[23]:


for i in data0.itertuples():
    mat=np.zeros((2,1))
    mat[0][0]=i.X
    mat[1][0]=i.Y
    val=mat[0][0]*direction[0][0]+mat[1][0]*direction[1][0]
#     val=val/direc
    proj_zero.append(val)
#     c=np.dot(direction,direction)


# In[24]:


# print(proj_zero)


# In[25]:


proj_one=[]    ##finding projection of class zero element on line
for i in data1.itertuples():
    mat=np.zeros((2,1))
    mat[0][0]=i.X
    mat[1][0]=i.Y
    val=mat[0][0]*direction[0][0]+mat[1][0]*direction[1][0]
#     val=val/direc
    proj_one.append(val)
#     c=np.dot(direction,direction)


# In[26]:


# print(proj_one)


# In[27]:



# x=data['X']
# y=data['Y']
# plt.scatter(x,y,c=data["target"])
for i in proj_zero:
    plt.scatter(i,0,color="r")
for i in proj_one:
    plt.scatter(i,0,color="b")
    


# In[28]:


val=mean0[0][0]*direction[0][0]+mean0[1][0]*direction[1][0]


mean0_proj=val
# print(mean0_proj)


# In[29]:


val=mean1[0][0]*direction[0][0]+mean1[1][0]*direction[1][0]

mean1_proj=val

mean=(mean1_proj+mean0_proj)/2

# print(mean)


# In[30]:


# for i in proj_zero:
#     plt.scatter(i,0,color="r")
# for i in proj_one:
#     plt.scatter(i,0,color="b")
# plt.scatter(mean,0,color="black",marker='*')
# print(mean)


# In[31]:


proj_one=np.array(proj_one);
proj_zero=np.array(proj_zero);

# print(proj_zero)
var0=np.var(proj_zero)
# print(var0)


# In[32]:


var1=np.var(proj_one)
# print(var1)


# In[33]:


meannew0=np.mean(proj_zero)
meannew1=np.mean(proj_one)
# print(meannew0)


# In[34]:


proj_one.sort()


# In[35]:


from scipy.stats import norm


# In[36]:


# proj_one


# In[37]:


# meannew1


# In[38]:


sigma_one = var1**0.5


# In[39]:


sigma_zero = var0**0.5


# In[40]:


# norm.pdf(proj_one, meannew1, sigma_one)


# In[41]:


X_axis_one = np.linspace(meannew1-4*sigma_one, meannew1+4*sigma_one,1000)


# In[42]:


X_axis_zer = np.linspace(meannew0-4*sigma_zero, meannew0+4*sigma_zero,1000)


# In[43]:


plt.plot(X_axis_one, norm.pdf(X_axis_one, meannew1, sigma_one))   ##plotting the normal curve
plt.plot(X_axis_zer, norm.pdf(X_axis_zer, meannew0, sigma_zero))


# In[44]:


# proj_zero.sort()


# In[45]:


# plt.figure(figsize=(10,6))

# plt.plot(proj_one, )

# plt.ylabel('gaussian distribution')
plt.show()

