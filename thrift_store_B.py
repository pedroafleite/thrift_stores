import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from altair import *
import pandas_profiling

#import Thrift Store B
thrift_b=pd.read_csv("C:/Users/Pedro/Documents/Python Scripts/Thrift stores/fixed typos/Thrift_Store_B.csv", sep=";", encoding='ISO-8859-1')

#create a new column for 'nomeDaPeca' by summing up multiple columns
thrift_b_nome = thrift_b[['PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']].copy()
thrift_b_nome2 = thrift_b_nome.stack().groupby(level=0).apply(' '.join).to_frame('nomeDaPeca')
print(thrift_b_nome2)

#Check whether there are compatible row numbers, then add it all to df as a new column
print(thrift_b.shape)
print(thrift_b_nome.shape)

print(thrift_b_nome)
thrift_b_nome2['nomeDaPeca'].values
thrift_b['nomeDaPeca'] = thrift_b_nome2['nomeDaPeca']

#rearranging columns to put 'nomeDaPeca' at 3rd place
thrift_b = thrift_b[['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']]
print(thrift_b.columns)
pd.DataFrame(thrift_b.dtypes)

#ONE LINE AND GET IT
report = pandas_profiling.ProfileReport(thrift_b) #spyder has some issues in displaying html
report.to_file('profile_report.html')

#tipos de roupas e acessórios
print("Roupas e acessórios: {}".format(thrift_b['PalavraChave1']))

#Comparar preços totais e com desconto
thrift_b.plot(kind='scatter', x='precoComDesconto', y='precoSemDesconto', color='red')
fig = sns.scatterplot(x='precoComDesconto', y='precoSemDesconto', hue='Disponível', data=thrift_b)
#The main trend that is seen is equivalent to 25% of discount

#Histogram to better show the wholeness of data
thrift_b[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,500])
thrift_b[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,40])

#WordCloud para Descriçao
descricao_text = str(thrift_b.Descricao)
wordcloud = WordCloud(width=480, height=480, margin=0, stopwords=['da', 'de', 'cu', 'length', 'com', 'fe', 'bol']).generate(descricao_text)
plt.imshow(wordcloud, interpolation='bilinear')

#WordCloud para PalavraChave1
PalavraChave1_text = str(thrift_b.PalavraChave1)
wordcloud2 = WordCloud(width=480, height=480, margin=0, stopwords=['Name', 'Length','dtype', 'PalavraChave1', 'object']).generate(PalavraChave1_text)
plt.imshow(wordcloud2, interpolation='bilinear')

#How much of each thing do you have?
thrift_b.groupby('PalavraChave1')['id'].count()
#See all the rows. 1) Change your buffer to 2000, then:
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', None)
#makesure to change it back

#Count and order by decreasing number of items. 
#To do that, slice a series and sort the series
thrift_b.groupby('PalavraChave1')['PalavraChave1'].count().nlargest(20)

#Histogram now!
color = plt.cm.winter(np.linspace(0, 10, 100))
x = pd.DataFrame(thrift_b.groupby(['PalavraChave1'])['id'].sum().reset_index())
x.sort_values(by = ['id'], ascending = False, inplace = True)

sns.barplot(x['PalavraChave1'].head(10), y = x['id'].head(10), data = x, palette = 'winter')
plt.title('Top 10 Countries in Suicides', fontsize = 20)
plt.xlabel('Name of Country')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()

#Divide one column for the other to obtain % of discount
thrift_b['Desconto'] = (thrift_b['precoComDesconto']/thrift_b['precoSemDesconto'])*100
thrift_b[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
#Show only Vestido
subset = thrift_b[thrift_b.PalavraChave1=='Óculos']
subset[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])



#Convert object to category
for col in ['PalavraChave1']: thrift_b[col] = thrift_b[col].astype('category')
print(thrift_b.PalavraChave1)

for col in ['Disponível']: thrift_b[col] = thrift_b[col].astype('category')
print(thrift_b.Disponível)

#Histogram, scatterplot to category using seaborn
print(thrift_b.columns)
pd.DataFrame(thrift_b.dtypes)
df = thrift_b
x1 = thrift_b.loc[thrift_b.PalavraChave1=='Vestido', 'precoComDesconto']
x2 = thrift_b.loc[thrift_b.PalavraChave1=='Óculos', 'precoComDesconto']
x3 = thrift_b.loc[thrift_b.PalavraChave1=='Tênis', 'precoComDesconto']
x4 = thrift_b.loc[thrift_b.PalavraChave1=='Top', 'precoComDesconto']
x5 = thrift_b.loc[thrift_b.PalavraChave1=='Suéter', 'precoComDesconto']
x6 = thrift_b.loc[thrift_b.PalavraChave1=='Shorts', 'precoComDesconto']
x7 = thrift_b.loc[thrift_b.PalavraChave1=='Sapato', 'precoComDesconto']
x8 = thrift_b.loc[thrift_b.PalavraChave1=='Sapatilha', 'precoComDesconto']
x9 = thrift_b.loc[thrift_b.PalavraChave1=='Sandália', 'precoComDesconto']
x10 = thrift_b.loc[thrift_b.PalavraChave1=='Saia', 'precoComDesconto']

kwargs = dict(alpha=0.5, bins=100)
plt.hist(x1, **kwargs, color='g', label='Vestido')
plt.hist(x2, **kwargs, color='b', label='Óculos')
plt.hist(x3, **kwargs, color='r', label='Tênis')
plt.hist(x10, **kwargs, color='y', label='Saia')
plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')

fig = sns.scatterplot(x='precoComDesconto', y='precoSemDesconto', hue='Disponível', data=thrift_b)

#Histogram, scatterplot to category using matplotlib
x=thrift_b['precoComDesconto']
y=thrift_b['precoSemDesconto']
uniq=list(set(thrift_b['Disponível']))
z = range(1,len(uniq))
hot = plt.get_cmap('hot')
cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

#Histogram, scatterplot to category using altair
