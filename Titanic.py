#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
df = pd.read_csv('train.csv') 


# ## Görev 1:  Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
# 

# In[241]:


df = sns.load_dataset("titanic")


# ## Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

# In[242]:


df.columns


# In[243]:


df["sex"].value_counts()


# ## Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

# In[244]:


df.nunique()


# ## Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

# In[219]:


print(df["pclass"].nunique())
print(df["pclass"].value_counts())


# ## Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

# In[220]:


df[["pclass","parch"]].nunique()
     


# ## Görev 6: embarked değişkeninin tipini kontrol ediniz. Bu değişkenin tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

# In[221]:


df["embarked"].dtype


# In[222]:


df["embarked"] = df["embarked"].astype("category")


# In[223]:


df["embarked"].info()


# ## Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.

# In[224]:


df.loc[df["embarked"]=="C"]


# ## Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

# In[225]:


df.loc[df["embarked"]!="S"]


# ## Görev 9:   Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

# In[226]:


df.loc[(df["age"]<30) & (df["sex"] == "female")]


# ## Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

# In[227]:


df.loc[(df["fare"]>500) | (df["age"]>70)] 


# ## Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

# In[228]:


df.isnull().sum()


# ## Görev 12: who değişkenini dataframe’ den çıkartınız.

# In[245]:


df = df.drop("who", axis = 1)


# In[246]:


df.head()


# ## Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

# In[247]:


df["deck"].unique()


# In[232]:


mode = df["deck"].mode().values[0]
print(mode)


# In[233]:


df.loc[df["deck"].isna() , 'deck'] = mode #deck'teki NaN değerleri en çok tekrar eden değer ile doldurma.


# In[234]:


df["deck"].unique()


# ## Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

# In[235]:


df["age"].unique()


# In[236]:


median = df["age"].median()
print(median)


# In[237]:


df.loc[df["age"].isna() , 'age'] = median


# In[238]:


df["age"].unique()


# ## Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

# In[239]:


df.head()


# In[240]:


df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]}) #agg fonksiyonu içinde belirterek hangi değişkenin hesaplanmasını istediğimizi belirtiyoruz.


# ## Görev 16: 30 yaşın altında olanlar 1, 30' a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz. 

# In[194]:


df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)


# In[195]:


df[["age","age_flag"]]


# ## Görev 17: Seaborn kütüphanesi içerisinden tips veri setini tanımlayınız.

# In[196]:


df = sns.load_dataset("tips")


# ## Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

# In[197]:


df.columns


# In[198]:


df["time"].unique()


# In[199]:


df.groupby(["time"]).agg({"total_bill":["sum","min","max","mean"]})


# ## Görev 19: Günler ve time' a göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

# In[200]:


df.columns


# In[201]:


df.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]})


# ## Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day' e göre toplamını, min, max ve ortalamasını bulunuz.

# In[202]:


df.head()


# In[203]:


df.loc[(df.time == "Lunch") & (df.sex == "Female")].groupby("day").agg({"total_bill":["mean","sum","max","min"], "tip":["mean","sum","max","min"]})


# In[204]:


df.columns


# ## Görev 21: size' ı 3' ten küçük, total_bill' i 10' dan büyük olan siparişlerin ortalaması nedir?

# In[205]:


df.head(6)


# In[206]:


df.loc[((df["size"] < 3) & (df["total_bill"] > 10)),["total_bill"]].mean() 


# ## Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Yeni oluşturulan değişken, her bir müşterinin ödediği totalbill ve tip' in toplamını verecek şekilde oluşturulmalıdır.

# In[207]:


df.head()


# In[208]:


df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]


# In[209]:


df.head(7)


# ## Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe' e atayınız.

# In[211]:


new_df = df.sort_values("total_bill_tip_sum", ascending = False).head(30)


# In[212]:


new_df

