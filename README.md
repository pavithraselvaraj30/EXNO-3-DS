## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![image](https://github.com/user-attachments/assets/70ecb9b3-7414-4088-b982-a1c1db0b10be)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/12f247cc-15e5-4792-832a-7a483381c93c)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/e91c4963-2780-478a-8621-2bf8f51c2f09)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/ccd9f3c0-61c5-4950-831f-13dacb1e0f43)
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/bd48bff2-b54b-443d-9082-cde2514df621)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/b66f3bc4-361a-41af-b357-ad0db0f6944c)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/c0f88649-425e-4a0e-b9ab-93d9d0270e4c)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```
![image](https://github.com/user-attachments/assets/33e4b50c-1bc5-4ecc-81dc-e7fdcb107410)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/9cd4ea7a-9640-43b2-b4d4-fedc94763ef0)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/8a5adafa-74c8-468b-a917-0e28fffddce7)
```
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![image](https://github.com/user-attachments/assets/4ab716bd-30da-47de-95b4-e608fb44aff0)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/3f100102-efb6-456a-b447-c397bb0cac02)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/82b585ac-f528-416d-bc12-faa2791d93c1)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ddaa32ff-e3d4-433d-aaf4-006c5a994c62)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0e601a34-3103-4cfb-9cb5-5b39b2a72a30)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4eec1e56-1ed1-4b48-bb0b-bae0e392e95b)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/2e8f30cc-d1c7-40d7-b6c7-57303799f017)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/4c4c1f19-ca5d-4aaa-9f38-d6b7194f8524)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/bfcaedb8-da17-4d80-af07-b1304d9109e1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/1a6686b3-4243-4343-b984-ba3a2208e03b)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/2a966293-f416-444e-b7c8-b443749c845c)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9d83e3ab-5dba-4449-b071-13cd3bacef74)
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/87764eb7-3d70-41b9-91f8-6b5c78f211ae)
```
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e9885c9c-3c14-481f-a21c-2e94457724c0)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/841cf4f1-bb94-4456-af13-f45cc45c0a58)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ae2960b7-8eea-4f2c-99c1-87ad43b9b7f8)

# RESULT:

   Feature Encoding and Transformation process has been successfully performed using the data set. 

 
