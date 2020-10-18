# Data Analysis and Strategy for Thrift Stores

### Table of contents
1. [Introduction and Data Description](#1-introduction-and-data-description)
    - 1.1 [What are our first impressions?](#11-what-are-our-first-impressions)
    
    
2. [Data Cleaning](#2-data-cleaning)
    - 2.1 [Thrift B](#21-thrift-b)
    - 2.2 [Thrift A and C](#22-thrift-a-and-c)
    
    
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
    - 3.1 [Thrift B](#31-thrift-b)
    - 3.2 [Thrift A and C](#32-thrift-a-and-c)
    - 3.3 [Conclusions of Exploratory Data Analysis](#33-conclusions-of-exploratory-data-analysis)
    - 3.4 [Recommendations for Future Datasets](#34-recommendations-for-future-datasets)
    
    
4. [Natural Language Processing (NLP)](#4-natural-language-processing-nlp)
    - 4.1 [Wordcloud](#41-wordcloud)
    - 4.2 [Bag-of-Words](#42-bag-of-words)
       - 4.2.1 [Defining a new categorical variable](#421-defining-a-new-categorical-variable)
       - 4.2.2 [Training and Testing Data](#422-training-and-testing-data)
       - 4.2.3 [Tokenization](#423-tokenization)

## 1. Introduction and Data Description
 
Our objective here is to perform a statistical analysis that will help a newly opened thrift store to make efficient and profitful businesses decisions. 

That are three different Thrift Stores: A, B and C. 

For didactic purposes, the abovementioned steps will be somewhat shuffled midway to give the logical perspective of the data analysis. Firstly, we will clear the data and explore the Thrift Store B dataset. Only after going through Thrift Store B, we will go for Thrift Stores A and C. Then, we will evaluate the data holistically and take (1) broad conclusions for the thrift store market and (2) specific conclusions for each thrift store.


```python
#Import libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels
import mglearn
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from altair import *
```

The dataset does not have a `README.txt` file, so we will have to infer most of the meaning of the columns.

Let's have a quick overview at all three thrift stores, to see if their components and columns match. Let' also check whether the datasets can be integrated with each other.


```python
#import Thrift Store A
thrift_a=pd.read_csv("C:/Users/Pedro/Documents/Python Scripts/Thrift stores/fixed typos/Thrift_Store_A.csv", sep=";", encoding='ISO-8859-1')
print(thrift_a)
```

              id                marca                nomeDaPeca  precoComDesconto  \
    0      21244  D&g Dolce & Gabbana      Óculos Lente Azulada             320.0   
    1       9981             Givenchy          Bolsa Coral Saco            2050.0   
    2      84176            Joe Fresh       Camisa Xadrez Verde              63.0   
    3      47475                Mixed   Calça Alfaiataria Preta             140.0   
    4      74864                 Zara     Vestido Jeans Babados              60.0   
    ...      ...                  ...                       ...               ...   
    56656   6109                Mixed        Camiseta Bothanica             207.0   
    56657  70968                Mixed          Blusa Seda Preta             190.0   
    56658  86400       Toca do Coelho      Body Verde Palmeiras              40.0   
    56659  87138         Bazar Genial  Babador Atoalhado Branco              12.0   
    56660   9561                 Farm     Saia Recortes Estampa              75.0   
    
           precoSemDesconto                                          Descricao  \
    0                1120.0  Óculos preto. Possui uma lente só em degradê. ...   
    1                4000.0  Bolsa coral de couro. Possui costuras aparente...   
    2                 310.0  Camisa xadrez nas cores verde e branco. Possui...   
    3                 490.0  Calça de alfaiataria na cor preta. Possui mode...   
    4                 130.0  Vestido jeans em lavagem clara. Possui modelag...   
    ...                 ...                                                ...   
    56656             690.0  Blusinha preta. com gola arredondada; Estampa ...   
    56657             620.0  Blusa na cor preta. Possui manga de renda com ...   
    56658               NaN  Body na cor verde estampa modelagem. Possui mo...   
    56659               NaN  Babador na cor branca. Possuí modelagem redond...   
    56660             225.0  Saia mini com estampa multicolorida. Possui de...   
    
          Tamanho           Condicao               Cores  Disponível     Data  \
    0           U              Usado          \nPreto \n     VENDIDO  28:36.5   
    1           U              Usado          \nCoral \n         NaN  42:03.6   
    2           P              Usado          \nVerde \n         NaN  44:38.9   
    3          34              Usado          \nPreto \n         NaN  50:45.9   
    4           P              Usado          \nJeans \n     VENDIDO  51:04.5   
    ...       ...                ...                 ...         ...      ...   
    56656       G  Novo com etiqueta  \nPreto \nVerde \n     VENDIDO  15:39.7   
    56657       M              Usado          \nPreto \n  DISPONIVEL  15:41.2   
    56658      3M              Usado                 NaN     VENDIDO  41:18.1   
    56659       U              Usado         \nBranco \n  DISPONIVEL  15:43.0   
    56660       G              Usado  \nMulticolorido \n     VENDIDO  15:44.0   
    
               Status  
    0      VERIFICADO  
    1      VERIFICADO  
    2      VERIFICADO  
    3      VERIFICADO  
    4      VERIFICADO  
    ...           ...  
    56656  VERIFICADO  
    56657  VERIFICADO  
    56658  VERIFICADO  
    56659  VERIFICADO  
    56660  VERIFICADO  
    
    [56661 rows x 12 columns]
    


```python
#import Thrift Store B
thrift_b=pd.read_csv("C:/Users/Pedro/Documents/Python Scripts/Thrift stores/fixed typos/Thrift_Store_B.csv", sep=";", encoding='ISO-8859-1')
print(thrift_b)
```

              id        marca  precoComDesconto  precoSemDesconto  \
    0      24427       nativa             49.99            100.00   
    1      60509          mob            139.97            449.90   
    2      42602         farm             74.70            249.00   
    3      41552   pure knite             25.25            135.00   
    4      60624         zara             26.21            189.90   
    ...      ...          ...               ...               ...   
    60290  50079  clock house             27.20             80.00   
    60291  38714        magia             21.48             85.90   
    60292  41740         zara             27.00             89.99   
    60293  35819      oshkosh             10.00             29.99   
    60294  74181   forever 21             29.44            117.75   
    
                                                   Descricao Tamanho  \
    0      Sandália anabela de tecido laminado vazado, fe...      37   
    1      Vestido camisa xadrez, com mangas 7/8. Bolsos ...       m   
    2      Vestido Farm cor verde militar, comprimento cu...       p   
    3      Colete em crochê, cor marrom claro, modelo com...       m   
    4      Vestido reto de tecido plano laranja escuro, c...      pp   
    ...                                                  ...     ...   
    60290  Calça jeans com modelagem skinny, com dois bol...      44   
    60291  Vestido longo, com estampa floral colorida, da...       m   
    60292  Camisa branca, com bolso único na regiÃ£o do t...     18m   
    60293  Camisetinha raglan com decote fechado, manga c...       3   
    60294  Blusa feminina , verde, da Forever 21. Tecido ...       m   
    
                    Condicao     Cores                   Composicao  \
    0      gentilmente usada   laranja              Tecido laminado   
    1      gentilmente usada  colorido        55% linho 45% algodão   
    2      nova com etiqueta     verde               100% poliéster   
    3      gentilmente usada    marrom                 sem etiqueta   
    4      gentilmente usada   laranja               100% poliéster   
    ...                  ...       ...                          ...   
    60290  gentilmente usada      azul                 sem etiqueta   
    60291  gentilmente usada  colorido   87% poliamida 13% elastano   
    60292  gentilmente usada    branco       80% AlgodÃ£o 20% Linho   
    60293  gentilmente usada  colorido  85% poliéster, 15% elastano   
    60294  gentilmente usada     verde               100% Poliéster   
    
                                                     Medidas  ...     Data  \
    0                                           Salto - 12cm  ...  13:33.0   
    1      Busto: 48 cm Cintura: 44 cm Quadril: 48 cm Com...  ...  13:37.1   
    2      Busto: 40 cm Cintura: 32 cm Quadril: 42 cm Com...  ...  13:38.3   
    3             Busto: 53 cm Cintura: 60 cm Quadril: 68 cm  ...  13:40.4   
    4      Busto: 44 cm Cintura: 46 cm Quadril: 50 cm Com...  ...  13:42.0   
    ...                                                  ...  ...      ...   
    60290  Cintura: 44 cm Quadril: 52 cm Comprimento: 107 cm  ...  05:27.1   
    60291  Busto: 39 cm Cintura: 37 cm Quadril: 59 cm Com...  ...  05:32.7   
    60292     Busto: 30 cm Cintura: 30 cm Comprimento: 35 cm  ...  05:41.9   
    60293                Busto 28cm, cintura 26cm, comp 38cm  ...  05:44.4   
    60294  Busto: 51 cm Cintura: 54 cm Quadril: 56 cm Com...  ...  05:47.0   
    
               Status PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4  \
    0      VERIFICADO       Anabela          Laço       Laranja                 
    1      VERIFICADO       Vestido        Camisa        Xadrez           Mob   
    2      VERIFICADO       Vestido         Verde       Militar          Farm   
    3      VERIFICADO        Colete            de        Crochê        Marrom   
    4      VERIFICADO       Vestido       Laranja           com      Bolinhas   
    ...           ...           ...           ...           ...           ...   
    60290  VERIFICADO         Calça         Jeans        Skinny         Clock   
    60291  VERIFICADO       Vestido         Longo        Floral         Magia   
    60292  VERIFICADO        Camisa        Branca          Zara                 
    60293  VERIFICADO      Camiseta      Infantil        Raglan                 
    60294  VERIFICADO         Blusa         Verde       Forever            21   
    
          PalavraChave5 PalavraChave6 PalavraChave7 PalavraChave8  
    0                                                              
    1                                                              
    2                                                              
    3             Claro                                            
    4             Azuis                                            
    ...             ...           ...           ...           ...  
    60290         House                                            
    60291                                                          
    60292                                                          
    60293                                                          
    60294                                                          
    
    [60295 rows x 21 columns]
    


```python
#Import Thrift Store C
thrift_c=pd.read_csv("C:/Users/Pedro/Documents/Python Scripts/Thrift stores/fixed typos/Thrift_Store_C.csv", sep=";", encoding='ISO-8859-1')
print(thrift_c)
```

             id           marca                     nomeDaPeca  precoComDesconto  \
    0     22701       Taverniti        Jaqueta Jeans Taverniti             199.0   
    1     22703       Sem Marca      Camisa Estampada Colorida             129.0   
    2     22705   Robert Pierre          Bermuda Robert Pierre              49.0   
    3     22442  Croft & Barrow  Camisa Vintage Croft & Barrow             129.0   
    4     22443          Boo Jo          Camisa Vintage Boo Jo             129.0   
    ...     ...             ...                            ...               ...   
    1915  14291         Lay Out                 Camisa Lay Out              25.0   
    1916  22486         Puritan      Camisa Vermelha Estampada             129.0   
    1917  22491    Falls Creack    Camisa Vintage Falls Creeck             129.0   
    1918  22496       Sem Marca         Blusão Bomber Colorido             199.0   
    1919  22498       Sem Marca     Bomber Jacket Candy Colors             249.0   
    
          precoSemDesconto                                          Descricao  \
    0                199.0  Jaqueta oversized jeans escuro, garimpada em B...   
    1                129.0  Camisa vintage com estampa colorida, de manga ...   
    2                 49.0  Bermuda jeans cinza, vintage, com cinco bolsos...   
    3                129.0  Camisa vintage com estampa tropical, garimpada...   
    4                129.0  Camisa vintage estampada, garimpada em Buenos ...   
    ...                ...                                                ...   
    1915              40.0  Camisa estampada de manga comprida com um bols...   
    1916             129.0  Camisa vintage, garimpada em Buenos Aires, Arg...   
    1917             129.0  Camisa vintage, garimpada em Buenos Aires, Arg...   
    1918             199.0  Blusão estilo bomber, com recortes coloridos. ...   
    1919             249.0  Jaqueta bomber oversized, com recortes colorid...   
    
         Tamanho Condicao  Cores  Disponível     Data      Status  
    0          M  Vintage    NaN  DISPONIVEL  46:38.3  VERIFICADO  
    1          G  Vintage    NaN  DISPONIVEL  49:11.0  VERIFICADO  
    2         42  Vintage    NaN  DISPONIVEL  46:21.3  VERIFICADO  
    3          G  Vintage    NaN  DISPONIVEL  46:55.0  VERIFICADO  
    4          M  Vintage    NaN  DISPONIVEL  49:27.8  VERIFICADO  
    ...      ...      ...    ...         ...      ...         ...  
    1915       G  Vintage    NaN     VENDIDO  44:54.9  VERIFICADO  
    1916       G  Vintage    NaN     VENDIDO  22:29.3  VERIFICADO  
    1917       G  Vintage    NaN     VENDIDO  58:35.5  VERIFICADO  
    1918       M  Vintage    NaN     VENDIDO  24:13.7  VERIFICADO  
    1919       G  Vintage    NaN     VENDIDO  42:27.9  VERIFICADO  
    
    [1920 rows x 12 columns]
    


```python
#Display column names
print(thrift_a.columns)
print(thrift_b.columns)
print(thrift_c.columns)
```

    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Disponível', 'Data',
           'Status'],
          dtype='object')
    Index(['id', 'marca', 'precoComDesconto', 'precoSemDesconto', 'Descricao',
           'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível',
           'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3',
           'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7',
           'PalavraChave8'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Disponível', 'Data',
           'Status'],
          dtype='object')
    

### 1.1 What are our first impressions?
                                                                
- `id`: Number of identification.                      
- `marca`: *Brand*.
- `nomeDaPeca`: *Name of the piece of clothing*. It is a full sentence describing in a few words the pieces of clothing in Thrift Stores A and C. Thrift Store B dataset is lacking the `nomeDaPeca` field. 
- `precoComDesconto`: *Discount price*. Currency in reais.
- `precoSemDesconto`: *Full price (without discount)*. Currency in reais.
- `Descricao`: *Description*. Detailed description of the object being sold. It might contain whether it has pockets, where it was brought, some measuments, whether there are defects in the piece, what kind of cloth it was made of, and so on.
- `Tamanho`: *Size*. In characters (PP, P, M, G, GG) and numbers (38, 39, 40...) depending on the type of clothing.
- `Condicao`: *Condition*. For Thrift Stores A and B, it states whether the piece is new or used, and if used, *how* used. In Thrift Store C, the whole column reads `Vintage` (which does not seem to be useful).
- `Cores`: *Colours*. In Thrift Store A, the column seems to correctly express the colour of the pieces. In Thrift Store B, the column seems to be correctly filled by the corresponding colour of the pieces of clothing. In Thrift Store C, the column is empty.
- `Disponível`: *Available.* It states whether the item is `DISPONíVEL` (available) or `VENDIDO` (sold).
- `Data`: *Date*. Turns out to be a non-sensical field, since it does not seem to refer to any common data terminology. It will not be useful.
- `Status`: For all items, it states a standard `VERIFICADO` (verified), probably meaning that the data input was inserted into the database.
- `PalavraChave`. *Keyword*. It apparently corresponds to the dismemberment of a full descriptive sentence regarding the product being sold, with the multiple instances of `PalavraChave1`, 2, 3, and so on, being able to be understood if fully merged in one sentence.


## 2. Data cleaning 
### 2.1 Thrift B

Multiple `PalavraChave`s (keywords) fields are present at Thrift Store B and the combination of these should be stored as a nomeDaPeca column.
These `PalavraChave`s (*keywords*) seem to aggregate to full sentences that would be properly placed under the "nomeDaPeca" field. In portuguese, nouns are generally the first word in the sentence, while adjectives come later. From these keywords, we can establish semantically that the first word of these keywords ("PalavraChave1") would be a noun. On the other hand, this syntax would not be possible in English, in which adjectives come first, and the noun varies as a first, second or third word; which would make our work harder.

Therefore, given that most of these points revolve around Thrift Store B, we will start from there.

It should be acknowledged that there were incongruencies given the special characters used in portuguese - such as á, ã, ç, ó, õ, é, and so on - that were changed to code in the original dataset. Since I'm a native portuguese speaker, I did a prior data cleaning on all datasets by fixing the special characters back to their accentuated forms using the "Find" tool in Excel. Other systematic typos were also corrected via Excel. 
On future analyses, I probably won't shy away from using Excel again for exploring the data at the beggining of the analysis, especially in cases such that the problem isn't so complex and the dataset is easy to survey.


```python
#create a new column for 'nomeDaPeca' by summing up multiple PalavraChave columns
thrift_b_nome = thrift_b[['PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']].copy()
thrift_b_nome2 = thrift_b_nome.stack().groupby(level=0).apply(' '.join).to_frame('nomeDaPeca')
print(thrift_b_nome)
print(thrift_b_nome2)
```

          PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5  \
    0           Anabela          Laço       Laranja                               
    1           Vestido        Camisa        Xadrez           Mob                 
    2           Vestido         Verde       Militar          Farm                 
    3            Colete            de        Crochê        Marrom         Claro   
    4           Vestido       Laranja           com      Bolinhas         Azuis   
    ...             ...           ...           ...           ...           ...   
    60290         Calça         Jeans        Skinny         Clock         House   
    60291       Vestido         Longo        Floral         Magia                 
    60292        Camisa        Branca          Zara                               
    60293      Camiseta      Infantil        Raglan                               
    60294         Blusa         Verde       Forever            21                 
    
          PalavraChave6 PalavraChave7 PalavraChave8  
    0                                                
    1                                                
    2                                                
    3                                                
    4                                                
    ...             ...           ...           ...  
    60290                                            
    60291                                            
    60292                                            
    60293                                            
    60294                                            
    
    [60295 rows x 8 columns]
                                         nomeDaPeca
    0                Anabela Laço Laranja          
    1             Vestido Camisa Xadrez Mob        
    2            Vestido Verde Militar Farm        
    3           Colete de Crochê Marrom Claro      
    4      Vestido Laranja com Bolinhas Azuis      
    ...                                         ...
    60290      Calça Jeans Skinny Clock House      
    60291        Vestido Longo Floral Magia        
    60292              Camisa Branca Zara          
    60293        Camiseta Infantil Raglan          
    60294            Blusa Verde Forever 21        
    
    [60295 rows x 1 columns]
    


```python
#Check whether these are compatible row numbers, so we can merge both dataframes
print(thrift_b.shape)
print(thrift_b_nome.shape)
```

    (60295, 22)
    (60295, 8)
    


```python
# Add "nomeDaPeca" as a new column
thrift_b_nome2['nomeDaPeca'].values
thrift_b['nomeDaPeca'] = thrift_b_nome2['nomeDaPeca']

#rearranging columns so that 'nomeDaPeca' can be at 3rd place, instead of the last
thrift_b = thrift_b[['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']]
print(thrift_b.columns)
```

    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas',
           'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2',
           'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6',
           'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    

### 2.2 Data cleaning: Thrift Stores A and C

Now that we have surveyed a bit of the Thrift B dataset, we shall evaluate Thrift Stores A and C.
It will be interesting to merge all three datasets for further analyses. To do this, we will have to create a similar-sized matrix for all datasets. Let's recapitulate the shape of each dataset to see what needs to be done.


```python
print(thrift_a.shape)
print(thrift_b.shape)
print(thrift_c.shape)

print(thrift_a.columns)
print(thrift_b.columns)
print(thrift_c.columns)

```

    (56661, 12)
    (60295, 22)
    (1920, 12)
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Disponível', 'Data',
           'Status'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas',
           'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2',
           'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6',
           'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Disponível', 'Data',
           'Status'],
          dtype='object')
    

Thrift Stores A and C lack all PalavraChave (keywords) fields. To fill those new columns, we shall dismember the nomeDaPeca (piece of clothing) field from each, in a process that is precisely the inverse of what we have done previously for the Thrift B dataset.


```python
#I would have to pick 'nomeDaPeca' and dismember it in multiple 'PalavraChaveX'.
new_list = thrift_a['nomeDaPeca'].apply(lambda x : pd.Series(x.split(' ')))
print(new_list)
```

                  0            1          2    3    4    5    6
    0        Óculos        Lente    Azulada  NaN  NaN  NaN  NaN
    1         Bolsa        Coral       Saco  NaN  NaN  NaN  NaN
    2        Camisa       Xadrez      Verde  NaN  NaN  NaN  NaN
    3         Calça  Alfaiataria      Preta  NaN  NaN  NaN  NaN
    4       Vestido        Jeans    Babados  NaN  NaN  NaN  NaN
    ...         ...          ...        ...  ...  ...  ...  ...
    56656  Camiseta    Bothanica        NaN  NaN  NaN  NaN  NaN
    56657     Blusa         Seda      Preta  NaN  NaN  NaN  NaN
    56658      Body        Verde  Palmeiras  NaN  NaN  NaN  NaN
    56659   Babador    Atoalhado     Branco  NaN  NaN  NaN  NaN
    56660      Saia     Recortes    Estampa  NaN  NaN  NaN  NaN
    
    [56661 rows x 7 columns]
    


```python
new_list2= thrift_c['nomeDaPeca'].apply(lambda x : pd.Series(x.split(' ')))
print(new_list2)
```

                0          1          2       3       4    5    6
    0     Jaqueta      Jeans  Taverniti     NaN     NaN  NaN  NaN
    1      Camisa  Estampada   Colorida     NaN     NaN  NaN  NaN
    2     Bermuda     Robert     Pierre     NaN     NaN  NaN  NaN
    3      Camisa    Vintage      Croft       &  Barrow  NaN  NaN
    4      Camisa    Vintage        Boo      Jo     NaN  NaN  NaN
    ...       ...        ...        ...     ...     ...  ...  ...
    1915   Camisa        Lay        Out     NaN     NaN  NaN  NaN
    1916   Camisa   Vermelha  Estampada     NaN     NaN  NaN  NaN
    1917   Camisa    Vintage      Falls  Creeck     NaN  NaN  NaN
    1918   Blusão     Bomber   Colorido     NaN     NaN  NaN  NaN
    1919   Bomber     Jacket      Candy  Colors     NaN  NaN  NaN
    
    [1920 rows x 7 columns]
    


```python
#Rename columns to match thrift__b
new_list.columns = ['PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7']
new_list2.columns = ['PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7']
print(new_list)
print(new_list2)
```

          PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5  \
    0            Óculos         Lente       Azulada           NaN           NaN   
    1             Bolsa         Coral          Saco           NaN           NaN   
    2            Camisa        Xadrez         Verde           NaN           NaN   
    3             Calça   Alfaiataria         Preta           NaN           NaN   
    4           Vestido         Jeans       Babados           NaN           NaN   
    ...             ...           ...           ...           ...           ...   
    56656      Camiseta     Bothanica           NaN           NaN           NaN   
    56657         Blusa          Seda         Preta           NaN           NaN   
    56658          Body         Verde     Palmeiras           NaN           NaN   
    56659       Babador     Atoalhado        Branco           NaN           NaN   
    56660          Saia      Recortes       Estampa           NaN           NaN   
    
          PalavraChave6 PalavraChave7  
    0               NaN           NaN  
    1               NaN           NaN  
    2               NaN           NaN  
    3               NaN           NaN  
    4               NaN           NaN  
    ...             ...           ...  
    56656           NaN           NaN  
    56657           NaN           NaN  
    56658           NaN           NaN  
    56659           NaN           NaN  
    56660           NaN           NaN  
    
    [56661 rows x 7 columns]
         PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5  \
    0          Jaqueta         Jeans     Taverniti           NaN           NaN   
    1           Camisa     Estampada      Colorida           NaN           NaN   
    2          Bermuda        Robert        Pierre           NaN           NaN   
    3           Camisa       Vintage         Croft             &        Barrow   
    4           Camisa       Vintage           Boo            Jo           NaN   
    ...            ...           ...           ...           ...           ...   
    1915        Camisa           Lay           Out           NaN           NaN   
    1916        Camisa      Vermelha     Estampada           NaN           NaN   
    1917        Camisa       Vintage         Falls        Creeck           NaN   
    1918        Blusão        Bomber      Colorido           NaN           NaN   
    1919        Bomber        Jacket         Candy        Colors           NaN   
    
         PalavraChave6 PalavraChave7  
    0              NaN           NaN  
    1              NaN           NaN  
    2              NaN           NaN  
    3              NaN           NaN  
    4              NaN           NaN  
    ...            ...           ...  
    1915           NaN           NaN  
    1916           NaN           NaN  
    1917           NaN           NaN  
    1918           NaN           NaN  
    1919           NaN           NaN  
    
    [1920 rows x 7 columns]
    


```python
#Get another columns to be the same size as thrift_b
new_list["PalavraChave8"] = np.nan
new_list2["PalavraChave8"] = np.nan
print(new_list)
print(new_list2)
```

          PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5  \
    0            Óculos         Lente       Azulada           NaN           NaN   
    1             Bolsa         Coral          Saco           NaN           NaN   
    2            Camisa        Xadrez         Verde           NaN           NaN   
    3             Calça   Alfaiataria         Preta           NaN           NaN   
    4           Vestido         Jeans       Babados           NaN           NaN   
    ...             ...           ...           ...           ...           ...   
    56656      Camiseta     Bothanica           NaN           NaN           NaN   
    56657         Blusa          Seda         Preta           NaN           NaN   
    56658          Body         Verde     Palmeiras           NaN           NaN   
    56659       Babador     Atoalhado        Branco           NaN           NaN   
    56660          Saia      Recortes       Estampa           NaN           NaN   
    
          PalavraChave6 PalavraChave7  PalavraChave8  
    0               NaN           NaN            NaN  
    1               NaN           NaN            NaN  
    2               NaN           NaN            NaN  
    3               NaN           NaN            NaN  
    4               NaN           NaN            NaN  
    ...             ...           ...            ...  
    56656           NaN           NaN            NaN  
    56657           NaN           NaN            NaN  
    56658           NaN           NaN            NaN  
    56659           NaN           NaN            NaN  
    56660           NaN           NaN            NaN  
    
    [56661 rows x 8 columns]
         PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5  \
    0          Jaqueta         Jeans     Taverniti           NaN           NaN   
    1           Camisa     Estampada      Colorida           NaN           NaN   
    2          Bermuda        Robert        Pierre           NaN           NaN   
    3           Camisa       Vintage         Croft             &        Barrow   
    4           Camisa       Vintage           Boo            Jo           NaN   
    ...            ...           ...           ...           ...           ...   
    1915        Camisa           Lay           Out           NaN           NaN   
    1916        Camisa      Vermelha     Estampada           NaN           NaN   
    1917        Camisa       Vintage         Falls        Creeck           NaN   
    1918        Blusão        Bomber      Colorido           NaN           NaN   
    1919        Bomber        Jacket         Candy        Colors           NaN   
    
         PalavraChave6 PalavraChave7  PalavraChave8  
    0              NaN           NaN            NaN  
    1              NaN           NaN            NaN  
    2              NaN           NaN            NaN  
    3              NaN           NaN            NaN  
    4              NaN           NaN            NaN  
    ...            ...           ...            ...  
    1915           NaN           NaN            NaN  
    1916           NaN           NaN            NaN  
    1917           NaN           NaN            NaN  
    1918           NaN           NaN            NaN  
    1919           NaN           NaN            NaN  
    
    [1920 rows x 8 columns]
    


```python
#Now, add new_list and new_list2 as columns to thrift_a and thrift_c
print(thrift_a.shape)
print(thrift_c.shape)

print(new_list.shape)
print(new_list2.shape)

thrift_a = pd.concat([thrift_a,new_list], axis = 1)
thrift_c = pd.concat([thrift_c,new_list2], axis = 1)
print(thrift_a)
print(thrift_c)
```

    (56661, 12)
    (1920, 12)
    (56661, 8)
    (1920, 8)
              id                marca                nomeDaPeca  precoComDesconto  \
    0      21244  D&g Dolce & Gabbana      Óculos Lente Azulada             320.0   
    1       9981             Givenchy          Bolsa Coral Saco            2050.0   
    2      84176            Joe Fresh       Camisa Xadrez Verde              63.0   
    3      47475                Mixed   Calça Alfaiataria Preta             140.0   
    4      74864                 Zara     Vestido Jeans Babados              60.0   
    ...      ...                  ...                       ...               ...   
    56656   6109                Mixed        Camiseta Bothanica             207.0   
    56657  70968                Mixed          Blusa Seda Preta             190.0   
    56658  86400       Toca do Coelho      Body Verde Palmeiras              40.0   
    56659  87138         Bazar Genial  Babador Atoalhado Branco              12.0   
    56660   9561                 Farm     Saia Recortes Estampa              75.0   
    
           precoSemDesconto                                          Descricao  \
    0                1120.0  Óculos preto. Possui uma lente só em degradê. ...   
    1                4000.0  Bolsa coral de couro. Possui costuras aparente...   
    2                 310.0  Camisa xadrez nas cores verde e branco. Possui...   
    3                 490.0  Calça de alfaiataria na cor preta. Possui mode...   
    4                 130.0  Vestido jeans em lavagem clara. Possui modelag...   
    ...                 ...                                                ...   
    56656             690.0  Blusinha preta. com gola arredondada; Estampa ...   
    56657             620.0  Blusa na cor preta. Possui manga de renda com ...   
    56658               NaN  Body na cor verde estampa modelagem. Possui mo...   
    56659               NaN  Babador na cor branca. Possuí modelagem redond...   
    56660             225.0  Saia mini com estampa multicolorida. Possui de...   
    
          Tamanho           Condicao               Cores  Disponível     Data  \
    0           U              Usado          \nPreto \n     VENDIDO  28:36.5   
    1           U              Usado          \nCoral \n         NaN  42:03.6   
    2           P              Usado          \nVerde \n         NaN  44:38.9   
    3          34              Usado          \nPreto \n         NaN  50:45.9   
    4           P              Usado          \nJeans \n     VENDIDO  51:04.5   
    ...       ...                ...                 ...         ...      ...   
    56656       G  Novo com etiqueta  \nPreto \nVerde \n     VENDIDO  15:39.7   
    56657       M              Usado          \nPreto \n  DISPONIVEL  15:41.2   
    56658      3M              Usado                 NaN     VENDIDO  41:18.1   
    56659       U              Usado         \nBranco \n  DISPONIVEL  15:43.0   
    56660       G              Usado  \nMulticolorido \n     VENDIDO  15:44.0   
    
               Status PalavraChave1 PalavraChave2 PalavraChave3 PalavraChave4  \
    0      VERIFICADO        Óculos         Lente       Azulada           NaN   
    1      VERIFICADO         Bolsa         Coral          Saco           NaN   
    2      VERIFICADO        Camisa        Xadrez         Verde           NaN   
    3      VERIFICADO         Calça   Alfaiataria         Preta           NaN   
    4      VERIFICADO       Vestido         Jeans       Babados           NaN   
    ...           ...           ...           ...           ...           ...   
    56656  VERIFICADO      Camiseta     Bothanica           NaN           NaN   
    56657  VERIFICADO         Blusa          Seda         Preta           NaN   
    56658  VERIFICADO          Body         Verde     Palmeiras           NaN   
    56659  VERIFICADO       Babador     Atoalhado        Branco           NaN   
    56660  VERIFICADO          Saia      Recortes       Estampa           NaN   
    
          PalavraChave5 PalavraChave6 PalavraChave7  PalavraChave8  
    0               NaN           NaN           NaN            NaN  
    1               NaN           NaN           NaN            NaN  
    2               NaN           NaN           NaN            NaN  
    3               NaN           NaN           NaN            NaN  
    4               NaN           NaN           NaN            NaN  
    ...             ...           ...           ...            ...  
    56656           NaN           NaN           NaN            NaN  
    56657           NaN           NaN           NaN            NaN  
    56658           NaN           NaN           NaN            NaN  
    56659           NaN           NaN           NaN            NaN  
    56660           NaN           NaN           NaN            NaN  
    
    [56661 rows x 20 columns]
             id           marca                     nomeDaPeca  precoComDesconto  \
    0     22701       Taverniti        Jaqueta Jeans Taverniti             199.0   
    1     22703       Sem Marca      Camisa Estampada Colorida             129.0   
    2     22705   Robert Pierre          Bermuda Robert Pierre              49.0   
    3     22442  Croft & Barrow  Camisa Vintage Croft & Barrow             129.0   
    4     22443          Boo Jo          Camisa Vintage Boo Jo             129.0   
    ...     ...             ...                            ...               ...   
    1915  14291         Lay Out                 Camisa Lay Out              25.0   
    1916  22486         Puritan      Camisa Vermelha Estampada             129.0   
    1917  22491    Falls Creack    Camisa Vintage Falls Creeck             129.0   
    1918  22496       Sem Marca         Blusão Bomber Colorido             199.0   
    1919  22498       Sem Marca     Bomber Jacket Candy Colors             249.0   
    
          precoSemDesconto                                          Descricao  \
    0                199.0  Jaqueta oversized jeans escuro, garimpada em B...   
    1                129.0  Camisa vintage com estampa colorida, de manga ...   
    2                 49.0  Bermuda jeans cinza, vintage, com cinco bolsos...   
    3                129.0  Camisa vintage com estampa tropical, garimpada...   
    4                129.0  Camisa vintage estampada, garimpada em Buenos ...   
    ...                ...                                                ...   
    1915              40.0  Camisa estampada de manga comprida com um bols...   
    1916             129.0  Camisa vintage, garimpada em Buenos Aires, Arg...   
    1917             129.0  Camisa vintage, garimpada em Buenos Aires, Arg...   
    1918             199.0  Blusão estilo bomber, com recortes coloridos. ...   
    1919             249.0  Jaqueta bomber oversized, com recortes colorid...   
    
         Tamanho Condicao  Cores  Disponível     Data      Status PalavraChave1  \
    0          M  Vintage    NaN  DISPONIVEL  46:38.3  VERIFICADO       Jaqueta   
    1          G  Vintage    NaN  DISPONIVEL  49:11.0  VERIFICADO        Camisa   
    2         42  Vintage    NaN  DISPONIVEL  46:21.3  VERIFICADO       Bermuda   
    3          G  Vintage    NaN  DISPONIVEL  46:55.0  VERIFICADO        Camisa   
    4          M  Vintage    NaN  DISPONIVEL  49:27.8  VERIFICADO        Camisa   
    ...      ...      ...    ...         ...      ...         ...           ...   
    1915       G  Vintage    NaN     VENDIDO  44:54.9  VERIFICADO        Camisa   
    1916       G  Vintage    NaN     VENDIDO  22:29.3  VERIFICADO        Camisa   
    1917       G  Vintage    NaN     VENDIDO  58:35.5  VERIFICADO        Camisa   
    1918       M  Vintage    NaN     VENDIDO  24:13.7  VERIFICADO        Blusão   
    1919       G  Vintage    NaN     VENDIDO  42:27.9  VERIFICADO        Bomber   
    
         PalavraChave2 PalavraChave3 PalavraChave4 PalavraChave5 PalavraChave6  \
    0            Jeans     Taverniti           NaN           NaN           NaN   
    1        Estampada      Colorida           NaN           NaN           NaN   
    2           Robert        Pierre           NaN           NaN           NaN   
    3          Vintage         Croft             &        Barrow           NaN   
    4          Vintage           Boo            Jo           NaN           NaN   
    ...            ...           ...           ...           ...           ...   
    1915           Lay           Out           NaN           NaN           NaN   
    1916      Vermelha     Estampada           NaN           NaN           NaN   
    1917       Vintage         Falls        Creeck           NaN           NaN   
    1918        Bomber      Colorido           NaN           NaN           NaN   
    1919        Jacket         Candy        Colors           NaN           NaN   
    
         PalavraChave7  PalavraChave8  
    0              NaN            NaN  
    1              NaN            NaN  
    2              NaN            NaN  
    3              NaN            NaN  
    4              NaN            NaN  
    ...            ...            ...  
    1915           NaN            NaN  
    1916           NaN            NaN  
    1917           NaN            NaN  
    1918           NaN            NaN  
    1919           NaN            NaN  
    
    [1920 rows x 20 columns]
    


```python
#Fix the size of other columns as well. 
#There are other columns missing: Composição, Medidas and Desconto. Add empty columns for Composição and Medidas for now.
thrift_a["Composicao"] = np.nan
thrift_c["Composicao"] = np.nan
thrift_a["Medidas"] = np.nan
thrift_c["Medidas"] = np.nan

```


```python
#Thrift Stores A, B and C also needs a discount column.
thrift_a['Desconto'] = 100-((thrift_a['precoComDesconto']/thrift_a['precoSemDesconto'])*100)
thrift_b['Desconto'] = 100-((thrift_b['precoComDesconto']/thrift_b['precoSemDesconto'])*100)
thrift_c['Desconto'] = 100-((thrift_c['precoComDesconto']/thrift_c['precoSemDesconto'])*100)
```

    <ipython-input-46-a7a238428bce>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      thrift_b['Desconto'] = 100-((thrift_b['precoComDesconto']/thrift_b['precoSemDesconto'])*100)
    


```python
print(thrift_a.shape)
print(thrift_b.shape)
print(thrift_c.shape)

print(thrift_a.columns)
print(thrift_b.columns)
print(thrift_c.columns)
```

    (56661, 24)
    (60295, 24)
    (1920, 24)
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    


```python
#rearranging columns to make them right
thrift_a = thrift_a[['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto', 'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']]
print(thrift_a.columns)
thrift_b = thrift_b[['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto', 'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']]
print(thrift_b.columns)
thrift_c = thrift_c[['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto', 'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao', 'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1', 'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5', 'PalavraChave6', 'PalavraChave7', 'PalavraChave8']]
print(thrift_b.columns)
```

    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    


```python
#Let's merge the data from all thrift stores.
thrift_ab = thrift_a.append(thrift_b, ignore_index=True)
thrift_abc = thrift_ab.append(thrift_c, ignore_index=True)
print(thrift_a.shape)
print(thrift_b.shape)
print(thrift_c.shape)
print(thrift_ab.shape)
print(thrift_abc.shape)
print(thrift_abc.columns)
```

    (56661, 23)
    (60295, 23)
    (1920, 23)
    (116956, 23)
    (118876, 23)
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8'],
          dtype='object')
    

## 3. Exploratory Data Analysis
### 3.1 Thrift B

Now that all thrift store datasets are fixed, Thrift Store B has the nomeDaPeca (*piece of clothing*) column, similar to Thrift Stores A and C.
Let's analyse each Thrift Store individually, starting at Thrift B, while we are still on it.



```python
#Compare precoSemDesconto (price without discount) with precoComDesconto (price with discount)
thrift_b.plot(kind='scatter', x='precoComDesconto', y='precoSemDesconto', color='red')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x55248b0>




![png](output_26_1.png)


There are two main linear trends in the data. The diagonal trend seems to indicate a discount price that is about 70-75% lesser than the full price. For instance, at y(precoSemDesconto)=5000; while on x(precoComDesconto)≈1125-1500.
Let's explore this data further.


```python
#Add which itens are Disponível (available) or Vendido (sold)
fig = sns.scatterplot(x='precoComDesconto', y='precoSemDesconto', hue='Disponível', data=thrift_b)
```


![png](output_28_0.png)


This is interesting as well. "Vendido" means sold, while "Disponível" means available.
Pieces of clothing that have been on sale (with discount) seem to have a higher chance of being sold (which makes sense, especially at a 75% sellout as most pieces seem to be).

Let's check how these prices are distributed on a histogram:


```python
#Histogram to better show the wholeness of data
thrift_b[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,500])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x5598ac0>




![png](output_30_1.png)



```python
#Let's zoom in on the density peak of the precoComDesconto (Discount prices) to see the details of how the princes are arranged.
thrift_b[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,80])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x133de388>




![png](output_31_1.png)


Full prices (precoSemDesconto) are usually arranged at every 10 reais. Utilyzing "round numbers-1" may be a good sales pitch. That is, selling something for 79 reais instead of 80 reais. Or 89 instead of 90. Or 99 instead of 100. We are psychologycally more prone to relate a 99 reais price to a 90 reais rather than to 100 reais, even though the actual price is 9x closer to 100 reais rather than 90.
Also, discount prices also tend to cluster similarly to full prices (every 10 reais, buy with a 9 at as the second number). However, discount prices may also tend to occur at the fifths (25, 35, 45 reais, for instance).

Next step. Let's see which are the most popular types of clothing in Thrift Store B.


```python
#How much of each thing do you have
thrift_b.groupby('PalavraChave1')['id'].count()
```




    PalavraChave1
    .Camiseta      1
    Acessório      1
    Agasalho       5
    Alcinha       50
    Alcinhas       1
                ... 
    shorts         1
    vestido        1
    Ãculos        1
    Óculos       224
    óculos         2
    Name: id, Length: 463, dtype: int64




```python
#Count and order by decreasing number of items. 
#To do that, slice a series and sort it. Showing the 20 largest keyitems.
thrift_b_count = thrift_b.groupby('PalavraChave1')['PalavraChave1'].count().nlargest(20)
print(thrift_b_count)
```

    PalavraChave1
    Vestido     10494
    Calça        6645
    Blusa        5419
    Camisa       3976
    Saia         3920
    Blusinha     3443
    Camiseta     2652
    Shorts       1841
    Regata       1750
    Casaco       1030
    Bermuda       954
    Blazer        953
    Sandália      846
    Jaqueta       795
    Bolsa         762
    Body          716
    Suéter        680
    Sapato        679
    Bata          578
    Colete        551
    Name: PalavraChave1, dtype: int64
    

These are the top 20 pieces of clothing being sold (and having been sold) at Thrift Store B. However, some NLP will be required later to extract more information from this data. I trust that the interested reader will look for the portuguese words in a bilingual dictionary if there are any doubts on the meaning of each.


```python
#Assign a column filled with '1' to be further counted and grouped
thrift_b['item'] ='1'
thrift_a['item'] ='1' #do that for other thrift stores as well
thrift_c['item'] ='1'

print(thrift_a.columns)
print(thrift_b.columns)
print(thrift_c.columns)
```

    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item'],
          dtype='object')
    


```python
#Histogram the previous data
x = pd.DataFrame(thrift_b.groupby(['PalavraChave1'])['id'].sum().reset_index())
x.sort_values(by = ['id'], ascending = False, inplace = True)

sns.barplot(x['PalavraChave1'].head(10), y = (x['id'].head(10))*0.00002, data = x, palette = 'winter')
plt.title('Top 10 types of clothing', fontsize = 20)
plt.xlabel('Piece of clothing')
plt.xticks(rotation = 90)
plt.ylabel('Number of pieces')
plt.show()
```


![png](output_37_0.png)


Great! 
Let's go back to analysing discounts. 


```python
#Divide one column for the other to obtain % of discount
thrift_b[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13ae9e20>




![png](output_39_1.png)


Indeed, most discount seem to occur at the 70-75% range.
Now, let's see how the discount prices are distributed among specific pieces of clothing.


```python
#Show only Vestido (Dress)
subset_vestido = thrift_b[thrift_b.PalavraChave1=='Vestido']
subset_vestido[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
plt.xlabel('Vestido')

#Show only Calça (Trousers)
subset_calca = thrift_b[thrift_b.PalavraChave1=='Calça']
subset_calca[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
plt.xlabel('Calça')

#Show only Óculos (Spectacles)
subset_oculos = thrift_b[thrift_b.PalavraChave1=='Óculos']
subset_oculos[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
plt.xlabel('Óculos')

```




    Text(0.5, 0, 'Óculos')




![png](output_41_1.png)



![png](output_41_2.png)



![png](output_41_3.png)


Discount is given at about the same percentage rate regardless of which type of object is being sold.


```python
#Show only Vendidos (Sold)
subset_vendido = thrift_b[thrift_b.Disponível=='VENDIDO']
subset_vendido[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
plt.xlabel('Vendido')

subset_vendido = thrift_b[thrift_b.Disponível=='DISPONIVEL']
subset_vendido[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
plt.xlabel('Disponível')
```




    Text(0.5, 0, 'Disponível')




![png](output_43_1.png)



![png](output_43_2.png)


Discount also does not seem to be a factor for differing which pieces have been sold (Vendido) and which pieces are still available (Disponível). In fact, there seems to be a general proportion between objects sold and available, as Thrift Stores might reasonably showcase new products at about the same rate that the products from the former batch have been sold.

### 3.2 Thrift Stores A and C


```python
#Histogram to better show the wholeness of data
thrift_a[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,2000])
thrift_c[['precoSemDesconto', 'precoComDesconto']].plot(kind='hist', alpha=0.7, bins=200, range=[0,500])
#Thrift_a seems posher than others. And it seems to give the most amount of discounts. Let's see.
#On the other hand, on thrift_c, there is almost no difference between "Price with discount" and "without discount".
#Therefore, it is possible to merge all thrift_store data together; however, it would probably become a jumbled mess, since all three stores have different characteristics from each other

```




    <matplotlib.axes._subplots.AxesSubplot at 0x157b5718>




![png](output_46_1.png)



![png](output_46_2.png)



```python
thrift_a[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
thrift_c[['Desconto']].plot(kind='hist', alpha=0.7, bins=100, range=[0,100])
#Something is different on thrift_c.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x133cba60>




![png](output_47_1.png)



![png](output_47_2.png)



```python
#Lets analyse some real numbers
print('Thrift A')
print(thrift_a.describe())
print('\n')

print('Thrift B')
print(thrift_b.describe())
print('\n')

print('Thrift C')
print(thrift_c.describe())
print('\n')

print('Thrift A+B+C')
print(thrift_abc.describe())
```

    Thrift A
                      id  precoComDesconto  precoSemDesconto      Desconto  \
    count   56661.000000      56661.000000      54347.000000  54347.000000   
    mean    48153.907997        154.315458        505.941029     68.252088   
    std     28231.414985        295.597504        814.536897     18.083930   
    min      1957.000000          1.000000          0.400000  -3650.000000   
    25%     22717.000000         70.000000        224.500000     65.714286   
    50%     48276.000000        100.000000        340.000000     69.230769   
    75%     72721.000000        144.000000        490.000000     71.887550   
    max    102089.000000      23000.000000      27000.000000     99.821429   
    
           Composicao  Medidas  PalavraChave8  
    count         0.0      0.0            0.0  
    mean          NaN      NaN            NaN  
    std           NaN      NaN            NaN  
    min           NaN      NaN            NaN  
    25%           NaN      NaN            NaN  
    50%           NaN      NaN            NaN  
    75%           NaN      NaN            NaN  
    max           NaN      NaN            NaN  
    
    
    Thrift B
                     id  precoComDesconto  precoSemDesconto      Desconto
    count  60295.000000      60295.000000      60295.000000  60295.000000
    mean   49687.038196         39.406920        157.593594     71.734362
    std    19072.925914         68.758253        320.417815     11.102762
    min       20.000000          0.000000          1.310000   -663.358779
    25%    34449.500000         17.480000         69.000000     66.747229
    50%    50551.000000         25.900000         89.990000     70.823137
    75%    65863.500000         42.000000        160.000000     76.050268
    max    81779.000000       6604.750000      26419.000000    100.000000
    
    
    Thrift C
                     id  precoComDesconto  precoSemDesconto     Desconto  Cores  \
    count   1920.000000       1920.000000       1920.000000  1920.000000    0.0   
    mean   16241.843229         77.654687         83.331771     5.381767    NaN   
    std     2668.243838         55.395617         61.444772    14.867965    NaN   
    min    11687.000000          6.000000          6.000000     0.000000    NaN   
    25%    14534.500000         45.000000         49.000000     0.000000    NaN   
    50%    15015.500000         59.000000         65.000000     0.000000    NaN   
    75%    17817.250000         89.000000         89.000000     0.000000    NaN   
    max    22736.000000        399.000000        399.000000    72.463768    NaN   
    
           Composicao  Medidas  PalavraChave8  
    count         0.0      0.0            0.0  
    mean          NaN      NaN            NaN  
    std           NaN      NaN            NaN  
    min           NaN      NaN            NaN  
    25%           NaN      NaN            NaN  
    50%           NaN      NaN            NaN  
    75%           NaN      NaN            NaN  
    max           NaN      NaN            NaN  
    
    
    Thrift A+B+C
                      id  precoComDesconto  precoSemDesconto       Desconto
    count  118876.000000     118876.000000     116562.000000  116562.000000
    mean    48416.104916         94.794621        318.787253      69.017798
    std     24126.338918        217.587248        627.052226      17.049194
    min        20.000000          0.000000          0.400000   -3650.000000
    25%     28898.750000         24.980000         89.900000      65.789474
    50%     49012.500000         52.480000        180.000000      70.000000
    75%     68105.000000        100.000000        354.000000      74.996596
    max    102089.000000      23000.000000      27000.000000     100.000000
    

On the descriptive statistics summary, tbe only fields that shall be considered are precoComDesconto (discount prices), precoSemDesconto (full prices) and Desconto (discount in %).


```python
#Let's analise only dresses (Vestido) now.
thrift_b_Vestido = thrift_b.loc[thrift_b.PalavraChave1=='Vestido']
thrift_b_Vestido

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>marca</th>
      <th>nomeDaPeca</th>
      <th>precoComDesconto</th>
      <th>precoSemDesconto</th>
      <th>Desconto</th>
      <th>Descricao</th>
      <th>Tamanho</th>
      <th>Condicao</th>
      <th>Cores</th>
      <th>...</th>
      <th>Status</th>
      <th>PalavraChave1</th>
      <th>PalavraChave2</th>
      <th>PalavraChave3</th>
      <th>PalavraChave4</th>
      <th>PalavraChave5</th>
      <th>PalavraChave6</th>
      <th>PalavraChave7</th>
      <th>PalavraChave8</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60509</td>
      <td>mob</td>
      <td>Vestido Camisa Xadrez Mob</td>
      <td>139.97</td>
      <td>449.90</td>
      <td>68.888642</td>
      <td>Vestido camisa xadrez, com mangas 7/8. Bolsos ...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Camisa</td>
      <td>Xadrez</td>
      <td>Mob</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42602</td>
      <td>farm</td>
      <td>Vestido Verde Militar Farm</td>
      <td>74.70</td>
      <td>249.00</td>
      <td>70.000000</td>
      <td>Vestido Farm cor verde militar, comprimento cu...</td>
      <td>p</td>
      <td>nova com etiqueta</td>
      <td>verde</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Verde</td>
      <td>Militar</td>
      <td>Farm</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60624</td>
      <td>zara</td>
      <td>Vestido Laranja com Bolinhas Azuis</td>
      <td>26.21</td>
      <td>189.90</td>
      <td>86.197999</td>
      <td>Vestido reto de tecido plano laranja escuro, c...</td>
      <td>pp</td>
      <td>gentilmente usada</td>
      <td>laranja</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Laranja</td>
      <td>com</td>
      <td>Bolinhas</td>
      <td>Azuis</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>63470</td>
      <td>msp</td>
      <td>Vestido Reto Listrado MSP</td>
      <td>32.50</td>
      <td>109.99</td>
      <td>70.451859</td>
      <td>Vestido feminino MSP . Tecido plano listrado c...</td>
      <td>38</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Reto</td>
      <td>Listrado</td>
      <td>MSP</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24337</td>
      <td>le ricard</td>
      <td>Vestido Godê Onça</td>
      <td>27.60</td>
      <td>69.90</td>
      <td>60.515021</td>
      <td>Vestido com estampa de onça, modelagem godê, r...</td>
      <td>p</td>
      <td>nova com etiqueta</td>
      <td>colorido</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Godê</td>
      <td>Onça</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60278</th>
      <td>78569</td>
      <td>zinzi</td>
      <td>Vestido Azul E Preto Zinzi</td>
      <td>23.70</td>
      <td>79.00</td>
      <td>70.000000</td>
      <td>Vestido, busto azul e saia preta, da marca Zin...</td>
      <td>p</td>
      <td>gentilmente usada</td>
      <td>azul</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Azul</td>
      <td>E</td>
      <td>Preto</td>
      <td>Zinzi</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>60279</th>
      <td>49105</td>
      <td>blue banana</td>
      <td>Vestido Preto Tomara que Caia</td>
      <td>152.00</td>
      <td>380.00</td>
      <td>60.000000</td>
      <td>Vestido com decote tomara que caia, modelagem ...</td>
      <td>p</td>
      <td>gentilmente usada</td>
      <td>preto</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Preto</td>
      <td>Tomara</td>
      <td>que</td>
      <td>Caia</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>60287</th>
      <td>69614</td>
      <td>pako's</td>
      <td>Vestido Pako's Malha Preta</td>
      <td>30.00</td>
      <td>69.90</td>
      <td>57.081545</td>
      <td>Vestido Pako's, de modelagem ampla, em malha p...</td>
      <td>p</td>
      <td>gentilmente usada</td>
      <td>preto</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Pako's</td>
      <td>Malha</td>
      <td>Preta</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>60289</th>
      <td>67762</td>
      <td>iódice</td>
      <td>Vestido Preto Iódice Drapeado</td>
      <td>63.21</td>
      <td>350.00</td>
      <td>81.940000</td>
      <td>Vestido Preto, de modelagem ampla, em malha ac...</td>
      <td>p</td>
      <td>gentilmente usada</td>
      <td>preto</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Preto</td>
      <td>Iódice</td>
      <td>Drapeado</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>60291</th>
      <td>38714</td>
      <td>magia</td>
      <td>Vestido Longo Floral Magia</td>
      <td>21.48</td>
      <td>85.90</td>
      <td>74.994179</td>
      <td>Vestido longo, com estampa floral colorida, da...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>VERIFICADO</td>
      <td>Vestido</td>
      <td>Longo</td>
      <td>Floral</td>
      <td>Magia</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10494 rows × 24 columns</p>
</div>




```python
thrift_b_Vestido['marca'].value_counts().head(30).plot(kind='barh', figsize=(20,10), fontsize=12)
plt.title('Thrift Store B - Vestido', fontsize = 20)
plt.ylabel('Marca')
plt.xlabel('Number of pieces')
plt.show()
```


![png](output_51_0.png)



```python
x2 = thrift_b_Vestido['marca'].value_counts().head(30)
print(x2)
```

    sem etiqueta      953
    zara              224
    farm              175
    forever 21        170
    sem marca         166
    hering            165
    c&a               123
    marisa            114
    le lis blanc      109
    renner             99
    antix              84
    h&m                78
    animale            75
    collins            68
    indefinida         67
    mercatto           64
    barred's           59
    luigi bertolli     56
    gregory            55
    maria filó         45
    linho fino         44
    mob                44
    lucidez            43
    ever pretty        42
    tvz                41
    zinzane            38
    shoulder           37
    m. officer         37
    riachuelo          35
    shop 126           34
    Name: marca, dtype: int64
    

Tags "sem etiqueta" and "sem marca" would mean "no tag" and "no brand" respectively.

Let's resume the objectives of this project: to perform a statistical analysis that will help businesses decisions in a newly opened thrift store.
 
### 3.3 Conclusions of Exploratory Data Analysis

- Thrift A displays a similar discount pattern to Thrift B, with most products ranging from 65-72%. Coincidentally or not, the median for Thrift Stores A discount prices is exactly 100 reais. It is also the most expensive store (A=100,00; B=25,90; C=59,00).
- Thrift B has the cheaper prices. It also has the most amounts of items in the dataset, which might indicate either that:
    - The sales have been well, probably prompted by the good prices, or that
    - Thrift Store B is just the oldest of all thrift store and it has been on the market for a longer period of time.
- Thrift C, on the other hand, has a different sales schema from the other thrift stores, in the sense that it barely puts discount on products. From our initial analysis, we saw that Thrift Store C may be somewhat more "hipster" than the former two thrift stores. It might have a different market strategy. Most pieces of clothing were purchased in Buenos Aires (Argentina). Also, Condicao (condition) field from Thrift Store C, instead of stating whether the piece of clothing is in a good state or not, it just says "Vintage" for all instances. "Vintage" is a term for trendy, oldfashioned clothes. This might indicate that Thrift Store C appeals to a very specific, smaller niche of costumers. Those characteristics do not necessarily indicate that the store is successful or not. It also does not mean that Thrift Store C is more expensive, since it actually has an intermediate price between Thrift Store B and A. But Thrift Store C indeed contains the least amount of items of all three stores, which means either that: 
    - The products showcased at the Thrift Store C were replaced at a lesser rate than in the other thrift stores, which might indicate that the sales haven't been so well, or that
    - Thrift Store C is relatively new, so it hasn't sold as many products as the other stores. 
    If the former supposition is the case, then Thrift Store C ends up being the least profitful of all Thrift Stores (which does not necessarily mean that it is the least successful, since success is completely relative to what your objectives are as an entrepreneur).
 
### 3.4 Recommendations for Future Datasets

The dataset would be considerably more useful, businesswise, if it had:
- The price that the Thrift Stores paid for each product before they were put on sale, which would allow us to calculate the profit.
- The correct dates for the price transactions.
- Customer data: especially gender and age.



## 4. Natural Language Processing (NLP)
We use some basic NLP techniques to extract more information from the Thrift Store datasets.


### 4.1 Wordcloud

First, Let's evaluate which were the most commonly used words in the description by exhibiting a "wordcloud", as a preview to our NLP.
We shall also eliminate common propositions in portuguese, and focus on nouns and adjectives that tend to better characterise the objects we are evaluating.


```python
#WordCloud para Descriçao (detailed description)
descricao_text = str(thrift_b.Descricao)
wordcloud = WordCloud(width=480, height=480, margin=0, stopwords=['da', 'do', 'de', 'cu', 'na','length', 'Descricao','Name','dtype','object','com', 'fe', 'bol', 'regiÃ']).generate(descricao_text)
plt.imshow(wordcloud, interpolation='bilinear')

```




    <matplotlib.image.AxesImage at 0x132edbb0>




![png](output_57_1.png)


Fine. Now let's apply wordcloud for the first order keyword (which would correspond to the main nouns in the dataset).


```python
#WordCloud para PalavraChave1
PalavraChave1_text = str(thrift_b.PalavraChave1)
wordcloud2 = WordCloud(width=480, height=480, margin=0, stopwords=['Name', 'Length','dtype', 'PalavraChave1', 'object']).generate(PalavraChave1_text)
plt.imshow(wordcloud2, interpolation='bilinear')
```




    <matplotlib.image.AxesImage at 0x158be5c8>




![png](output_59_1.png)


That does not seem to work well given that, intuitivelly, it is much harder to imagine that "Anabela" (a specific type of sandal) and "Colete" (waistcoat) would have the same weight as standard pieces of clothing such as "Calça" (trousers *aka in US* pants) and "Camisa" (shirt). 

Also, as we have seen previously, the most common pieces of clothing in Thrift Store B are: Vestido, Calça, Blusa, Camisa, Saia, Blusinha... And so on, which is incompatible, at least in this instance, with the results obtained via wordcloud.

### 4.2 Bag-of-Words

#### 4.2.1 Defining a new categorical variable
We will first establish a new categorical data to divide our data into "caro" (expensive) and "barato" (cheap). To define which one is which, we will first get the median for each dataset. In the real world, we would probably define which values are expensive or cheap depending on the values we have in our pocket. However, in our simulated world, we will artifically define this boundary so that we can simmetrically divide our datasets. The values above the median will be defined as expensive; the values below the median will be defined as cheap. 

Firstly, we will only analyse the data on Thrift Store B to avoid descriptive difference that would be inherent from the different stores we're modelling.
- Caro = 1
- Barato = 0


```python
#Let's create the new column and fill it with zeros
thrift_b["cost"] = 0
print(thrift_b.columns)
```

    Index(['id', 'marca', 'nomeDaPeca', 'precoComDesconto', 'precoSemDesconto',
           'Desconto', 'Descricao', 'Tamanho', 'Condicao', 'Cores', 'Composicao',
           'Medidas', 'Disponível', 'Data', 'Status', 'PalavraChave1',
           'PalavraChave2', 'PalavraChave3', 'PalavraChave4', 'PalavraChave5',
           'PalavraChave6', 'PalavraChave7', 'PalavraChave8', 'item', 'cost'],
          dtype='object')
    


```python
margin = 100
margin
```




    100




```python
#create a function that operates on the rows of the dataframe:
def f(row):
    if row['precoComDesconto'] >= 100:
        val = 1 #caro
    else:
        val = 0 #barato
    return val
```


```python
thrift_b['cost'] = thrift_b.apply(f, axis=1)
thrift_b
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>marca</th>
      <th>nomeDaPeca</th>
      <th>precoComDesconto</th>
      <th>precoSemDesconto</th>
      <th>Desconto</th>
      <th>Descricao</th>
      <th>Tamanho</th>
      <th>Condicao</th>
      <th>Cores</th>
      <th>...</th>
      <th>PalavraChave1</th>
      <th>PalavraChave2</th>
      <th>PalavraChave3</th>
      <th>PalavraChave4</th>
      <th>PalavraChave5</th>
      <th>PalavraChave6</th>
      <th>PalavraChave7</th>
      <th>PalavraChave8</th>
      <th>item</th>
      <th>cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24427</td>
      <td>nativa</td>
      <td>Anabela Laço Laranja</td>
      <td>49.99</td>
      <td>100.00</td>
      <td>50.010000</td>
      <td>Sandália anabela de tecido laminado vazado, fe...</td>
      <td>37</td>
      <td>gentilmente usada</td>
      <td>laranja</td>
      <td>...</td>
      <td>Anabela</td>
      <td>Laço</td>
      <td>Laranja</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60509</td>
      <td>mob</td>
      <td>Vestido Camisa Xadrez Mob</td>
      <td>139.97</td>
      <td>449.90</td>
      <td>68.888642</td>
      <td>Vestido camisa xadrez, com mangas 7/8. Bolsos ...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>Vestido</td>
      <td>Camisa</td>
      <td>Xadrez</td>
      <td>Mob</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42602</td>
      <td>farm</td>
      <td>Vestido Verde Militar Farm</td>
      <td>74.70</td>
      <td>249.00</td>
      <td>70.000000</td>
      <td>Vestido Farm cor verde militar, comprimento cu...</td>
      <td>p</td>
      <td>nova com etiqueta</td>
      <td>verde</td>
      <td>...</td>
      <td>Vestido</td>
      <td>Verde</td>
      <td>Militar</td>
      <td>Farm</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41552</td>
      <td>pure knite</td>
      <td>Colete de Crochê Marrom Claro</td>
      <td>25.25</td>
      <td>135.00</td>
      <td>81.296296</td>
      <td>Colete em crochê, cor marrom claro, modelo com...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>marrom</td>
      <td>...</td>
      <td>Colete</td>
      <td>de</td>
      <td>Crochê</td>
      <td>Marrom</td>
      <td>Claro</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60624</td>
      <td>zara</td>
      <td>Vestido Laranja com Bolinhas Azuis</td>
      <td>26.21</td>
      <td>189.90</td>
      <td>86.197999</td>
      <td>Vestido reto de tecido plano laranja escuro, c...</td>
      <td>pp</td>
      <td>gentilmente usada</td>
      <td>laranja</td>
      <td>...</td>
      <td>Vestido</td>
      <td>Laranja</td>
      <td>com</td>
      <td>Bolinhas</td>
      <td>Azuis</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60290</th>
      <td>50079</td>
      <td>clock house</td>
      <td>Calça Jeans Skinny Clock House</td>
      <td>27.20</td>
      <td>80.00</td>
      <td>66.000000</td>
      <td>Calça jeans com modelagem skinny, com dois bol...</td>
      <td>44</td>
      <td>gentilmente usada</td>
      <td>azul</td>
      <td>...</td>
      <td>Calça</td>
      <td>Jeans</td>
      <td>Skinny</td>
      <td>Clock</td>
      <td>House</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60291</th>
      <td>38714</td>
      <td>magia</td>
      <td>Vestido Longo Floral Magia</td>
      <td>21.48</td>
      <td>85.90</td>
      <td>74.994179</td>
      <td>Vestido longo, com estampa floral colorida, da...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>Vestido</td>
      <td>Longo</td>
      <td>Floral</td>
      <td>Magia</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60292</th>
      <td>41740</td>
      <td>zara</td>
      <td>Camisa Branca Zara</td>
      <td>27.00</td>
      <td>89.99</td>
      <td>69.996666</td>
      <td>Camisa branca, com bolso único na regiÃ£o do t...</td>
      <td>18m</td>
      <td>gentilmente usada</td>
      <td>branco</td>
      <td>...</td>
      <td>Camisa</td>
      <td>Branca</td>
      <td>Zara</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60293</th>
      <td>35819</td>
      <td>oshkosh</td>
      <td>Camiseta Infantil Raglan</td>
      <td>10.00</td>
      <td>29.99</td>
      <td>66.655552</td>
      <td>Camisetinha raglan com decote fechado, manga c...</td>
      <td>3</td>
      <td>gentilmente usada</td>
      <td>colorido</td>
      <td>...</td>
      <td>Camiseta</td>
      <td>Infantil</td>
      <td>Raglan</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60294</th>
      <td>74181</td>
      <td>forever 21</td>
      <td>Blusa Verde Forever 21</td>
      <td>29.44</td>
      <td>117.75</td>
      <td>74.997877</td>
      <td>Blusa feminina , verde, da Forever 21. Tecido ...</td>
      <td>m</td>
      <td>gentilmente usada</td>
      <td>verde</td>
      <td>...</td>
      <td>Blusa</td>
      <td>Verde</td>
      <td>Forever</td>
      <td>21</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>60295 rows × 25 columns</p>
</div>



### 4.2.2 Training and Testing Data


```python
#Finally, let's import sklearn
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))
```

    scikit-learn version: 0.23.1
    


```python
#Let's split the data into training and test sets
from sklearn.model_selection import train_test_split
text_train, text_test, y_train, y_test = train_test_split(thrift_b['Descricao'], thrift_b['cost'], random_state=0)
```


```python
print("text_train shape: {}".format(text_train.shape))
print("y_train shape: {}".format(y_train.shape))
```

    text_train shape: (45221,)
    y_train shape: (45221,)
    


```python
#The split usually puts 75% of data in the training data, and 25% at the test data
print("text_test shape: {}".format(text_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    text_test shape: (15074,)
    y_test shape: (15074,)
    


```python
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[1]:\n{}".format(text_train[1]))
```

    type of text_train: <class 'pandas.core.series.Series'>
    length of text_train: 45221
    text_train[1]:
    Vestido camisa xadrez, com mangas 7/8. Bolsos laterais, faixa na cintura e abotoamento simples. Tamanho M.
    


```python
print("Samples per class (training): {}".format(np.bincount(y_train)))
```

    Samples per class (training): [42773  2448]
    


```python
#Do the same for test dataset
print("type of text_test: {}".format(type(text_test)))
print("type of y_train: {}".format(type(y_train)))
print("type of y_test: {}".format(type(y_test)))
```

    type of text_test: <class 'pandas.core.series.Series'>
    type of y_train: <class 'pandas.core.series.Series'>
    type of y_test: <class 'pandas.core.series.Series'>
    

### 4.2.3 Tokenization


```python
#let's use a transformer called CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```


```python
#Transform to unicode string
X_train = vect.fit_transform(text_train.values.astype('U'))
print("X_train:\n{}".format(repr(X_train)))
```

    X_train:
    <45221x11783 sparse matrix of type '<class 'numpy.int64'>'
    	with 764555 stored elements in Compressed Sparse Row format>
    


```python
# Use get_feature_names method
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 1000 to 1030:\n{}".format(feature_names[1000:1030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))
```

    Number of features: 11783
    First 20 features:
    ['00', '000', '001', '00p', '01', '02', '03', '04', '05', '06', '08', '0p', '10', '100', '100cm', '100cmx', '100x100', '101', '101cm', '102']
    Features 1000 to 1030:
    ['apostando', 'aposte', 'apostrofe', 'apoá', 'apparel', 'apparels', 'apple', 'apresenta', 'apresentações', 'aprofundado', 'apropriada', 'apropriado', 'aproveita', 'aproveite', 'aproximadamente', 'apt9', 'aqua', 'aquamar', 'aquarela', 'aquario', 'aquela', 'aquele', 'ar', 'arabesco', 'arabescos', 'arabestos', 'aracaju', 'arallope', 'arame', 'aramell']
    Every 2000th feature:
    ['00', 'bunch', 'ellya', 'joshua', 'omni', 'silk']
    


```python
#Let's build a classifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
```

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Mean cross-validation accuracy: 0.95
    

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    


```python
#We obtained 95%, which is interesting
#Let's try to tune the C parameter of the LinearRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
```

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Best cross-validation score: 0.95
    Best parameters:  {'C': 1}
    

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    


```python
#generalised performance
print("{:.2f}".format(grid.score(X_train, y_train)))
```

    0.96
    


```python
X_test = vect.transform(text_test.values.astype('U'))
```


```python
#It's important to fit the X-test and y_test into the same grid

grid.fit(X_test, y_test)
```

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    GridSearchCV(cv=5, estimator=LogisticRegression(),
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10]})




```python
print("{:.2f}".format(grid.score(X_test, y_test)))
```

    0.95
    

So, that is what we've got! 75% in the test dataset. We certainly could improve the extraction of words.
Whenever we change the value that we divide `caro`and `barato`, we usually get an improvement in this value, until it starts to descend due to the lack of available samples.

So, when the value of the dataset is:
- 25.90 reais (which is the median) -> 75 %
- 100.00 reais -> 95 %

Further on, I could certainly write a pipeline to know how to distribute the values that I use to define which values are cheap or expensive.

It would be interesting to eliminate stopwords (that are words that appear too frequently but are not related to the meaning of the individual terms; for instance, prepositions). However, dictionary of stopwords in portuguese are not readily available, so we will have to skip this step for now.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
```


```python
pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

```


```python
grid = GridSearchCV(pipe, param_grid, cv=5)
text_train2 = text_train.fillna(' ')
grid.fit(text_train2, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
```

    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\Pedro\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Best cross-validation score: 0.95
    


```python
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("Features with lowest tfidf:\n{}".format(
feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(
feature_names[sorted_by_tfidf[-20:]]))
```

    Features with lowest tfidf:
    ['feminina' 'feminino' 'modelagem' 'acinturada' 'careca' 'ajustada'
     'compridas' 'reta' 'nadador' 'pockets' 'renner' 'suéter' 'five'
     'retangular' 'estampa' 'baixa' 'tecido' 'fixas' 'viscose' 'legging']
    Features with highest tfidf: 
    ['lindo' 'ou' 'cobre' 'uma' 'fita' 'tem' 'usado' 'compartimento' 'pérola'
     'regiã' 'lilás' 'mochila' 'camada' 'ilhóses' 'plissados' 'porta' 'tachas'
     'num' 'cm' 'velho']
    


```python
sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".format(
feature_names[sorted_by_idf[:100]]))
```

    Features with lowest idf:
    ['com' 'de' 'tecido' 'modelagem' 'em' 'da' 'fechamento' 'na' 'por'
     'decote' 'mangas' 'malha' 'zíper' 'plano' 'reta' 'bolsos' 'vestido' 'cor'
     'azul' 'estampa' 'no' 'modelo' 'cintura' 'feminina' 'frontal' 'sem'
     'calça' 'preta' 'gola' 'preto' 'abotoamento' 'curtas' 'elástico'
     'tamanho' 'nas' 'longas' 'parte' 'blusa' 'saia' 'dois' 'botões'
     'arredondado' 'botão' 'forro' 'frente' 'costas' 'cinza' 'detalhe' 'rosa'
     'comprimento' 'tom' 'camisa' 'jeans' 'verde' 'branca' 'blusinha'
     'feminino' 'branco' 'cós' 'manga' 'do' 'barra' 'alças' 'marrom' 'curto'
     'marca' 'lateral' 'busto' 'amarração' 'renda' 'etiqueta' 'um' 'evasê'
     'bege' 'laterais' 'camiseta' 'detalhes' 'regular' 'para' 'estampado'
     'canoa' 'peça' 'floral' 'colarinho' 'regata' 'invisível' 'metal' 'tons'
     'ampla' 'acinturada' 'colorida' 'salto' 'infantil' 'centro' 'shorts'
     'possui' 'careca' 'estampada' 'corpo' 'região']
    


```python
mglearn.tools.visualize_coefficients(
grid.best_estimator_.named_steps["logisticregression"].coef_,
feature_names, n_top_features=40)
```


![png](output_90_0.png)


This last diagram shows which brands are hierarchically related to cheap or expensive products.
The features were chosen by the algorhythm with the intent to best express the terms that are linked to each sample group.
Therefore, on the left end, with the red bars, we have the term "infantil" for children's clothes, and the terms "hering". "renner", "marisa" and "riachuelo" that are some of the biggest retail stores in Brazil. The fact that these terms are clustered towards the more cheap products makes sense.

On the other end, at the highest blue bars, we also have the clustering of more expensive brands. "Animale", according to our analysis, is the most expensive brand at Thrift Store B. 

"Chá" is not especifically related to any brand, since there are many that have "chá" in their name, such as "Rosa Chá", "Chá & Mel", "Chá Verde", and even as a colour description (for "Chá" means *tea*). Most of the weight of these features may be brought by the "Rosa Chá" brand that, for our knowledge, is indeed expensive. 

"Diesel" comes on third, and might refer to a very fashionable and sophisticated clothes colouring.

"Seda", which stands for *silk*, is a luxury product since Antiquity. Hence, it is well-represented in as the fourth most expensive-related term in our dataset.

And this concludes our analysis for now!
