#################################################
# WIKI 1 - Metin Ön İşleme ve Görselleştirme (NLP - Text Preprocessing & Text Visualization)
#################################################

#################################################
# Problemin Tanımı
#################################################
# Wikipedia örnek datasından metin ön işleme, temizleme işlemleri gerçekleştirip, görselleştirme çalışmaları yapmak.

#################################################
# Veri Seti Hikayesi
#################################################
# Wikipedia datasından alınmış metinleri içermektedir.

#################################################
# Gerekli Kütüphaneler ve ayarlar
#################################################

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# Datayı okumak
df = pd.read_csv("datasets/._wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape

#################################################
# Görevler:
#################################################

# Görev 1: Metindeki ön işleme işlemlerini gerçekleştirecek bir fonksiyon yazınız.
# •    Büyük küçük harf dönüşümünü yapınız.
# •    Noktalama işaretlerini çıkarınız.
# •    Numerik ifadeleri çıkarınız.

def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '')
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '')
    return text

df["text"] = clean_text(df["text"])

df.head()

# Görev 2: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak fonksiyon yazınız.

def remove_stopwords(text):
    stop_words = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])

# Görev 3: Metinde az tekrarlayan kelimeleri bulunuz.

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

# Görev 4: Metinde az tekrarlayan kelimeleri metin içerisinden çıkartınız. (İpucu: lambda fonksiyonunu kullanınız.)

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Görev 5: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words)

# Görev 6: Lemmatization işlemini yapınız.
# ran, runs, running -> run (normalleştirme)

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

# Görev 7: Metindeki terimlerin frekanslarını hesaplayınız. (İpucu: Barplot grafiği için gerekli)

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index() # kodu güncellemek gerekecek

tf.head()

# Görev 8: Barplot grafiğini oluşturunuz.

# Sütunların isimlendirilmesi
tf.columns = ["words", "tf"]
# 2000'den fazla geçen kelimelerin görselleştirilmesi
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Kelimeleri WordCloud ile görselleştiriniz.

# kelimeleri birleştirdik
text = " ".join(i for i in df["text"])

# wordcloud görselleştirmenin özelliklerini belirliyoruz
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 9: Tüm aşamaları tek bir fonksiyon olarak yazınız.
# •    Metin ön işleme işlemlerini gerçekleştiriniz.
# •    Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# •    Fonksiyonu açıklayan 'docstring' yazınız.

df = pd.read_csv("Modül_8_Dogal_Dil_İşleme/datasets/wiki_data.csv", index_col=0)

def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Metinler üzerinde ön işleme işlemleri yapar.

    :param text: DataFrame'deki metinlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: işlenmiş metin

    Örnek:
            wiki_preprocess(dataframe[col_name])
    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=False)
    # Numbers
    text = text.str.replace('\d', '', regex=True)
    # Stopwords
    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    if Barplot:
        # Terim Frekanslarının Hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sütunların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 2000'den fazla geçen kelimelerin görselleştirilmesi
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimeleri birleştirdik
        text_combined = " ".join(i for i in text)
        # wordcloud görselleştirmenin özelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_combined)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], Barplot=True, Wordcloud=True)
