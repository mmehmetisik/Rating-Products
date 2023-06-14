#################################################

# Bu bölümde satın alma da yaşanan ölçme problemlerine çözümler getireceğiz.
# Bir ürün satın alırken alıcının dikkat ettiği şeyler nerlerdir?
# - Ürün Puanları.
# - Ürün yıldız sayısı. (hangi yıldızdan kaç oy almış)
# - Ürün Yorum Sayısı.
# - Ürün satış adedi.
# - Sosyal ispat (Social Proof) (yani en faydalı yorum )
# İşte satıcının bir ürünü satarken, satın almasını etkileyen, pazar yeri olarak bu alıcıya en matıklı olan ürünü
# karşısına çıkarmak için yaptığımız ölçüm yöntemlerini inceleyecğiz.
# Kullanıcı: fiyat ve performans olarak en iyi ürüne ulaşmak istiyor. Biz de pazar yeri olarak bu konuda en iyi seçimi
# yapmasını sağlamak için bazı ölçüm yöntemlerini kullanarak tercih yapmasını kolaylaştırmak istiyoruz.
# Bir pazar yerinde ürünler sıralanırken aşağıdaki durumlar dikkate alınıyor.
# - Ürün Puanları
# - Ürünlerin yorum sayıları
# - Satın alma sayıları
# - Faydalı yorum sayısı

# Biz bu bölümde aşağıdaki yöntemleri kullanarak ölçüm problemlerini çözlmeye çalışacağız.
# - Rating Products
# - Sorting Products
# - Sorting Reviews
# - AB Testing
# Dynmaic Pricing

################################################



###################################################
# Rating Products - Ürün Puanlandırma
# => olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlandırma işlemlerini gerçekleştiriyor olacağız.
# Bu bölümde firmalerın websitelerinin ürünlerine verdiği puanları nasıl hesapladıklarını öğreneceğiz. Bu hesaplamaları
# aşağıdaki yöntemleri kullanacağız.

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

###################################################


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

# kullanacağımız kütüphaneleri programa dahil ettik.

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Burada bazı satır sütun ayarı yaptık.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("Miuul/Ölçüm Problemleri/datasets/course_reviews.csv")
df.head()
df.shape

# Veriyi biraz daha yakından tanımak adına bazı sorgulamalr yaparak veri hakkında daha fazla bilgi edinmeye çalışalım.
# rating dagılımı
df["Rating"].value_counts() # yani bu ürüne verilen puanarın dağılımına bakıyoruz.

df["Questions Asked"].value_counts() # bu ürüne ile ilgili sorulan soruların dağılımına bakıyoruz.

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})
# yukarıda sorulan soru özelinde kırılım yapılılarak agg ile bu kırılımda soru soranın sayısı ve puan ortalamasına
# ulaştık.

#                 Questions Asked  Rating
# Questions Asked
# 0.00000                     3867 4.76519 => burad şunu gördük hiç soru sormayanların sayısı 3867 ve puan ortalaması 4.76519
# 1.00000                      276 4.74094 => 1 soru soru soran 276 puan ortalaması 4.74094
# 2.00000                       80 4.80625
# 3.00000                       43 4.74419
# 4.00000                       15 4.83333
# 5.00000                       13 4.65385
# 6.00000                        9 5.00000
# 7.00000                        2 4.75000
# 8.00000                        5 4.90000
# 9.00000                        3 5.00000
# 10.00000                       2 5.00000
# 11.00000                       2 5.00000
# 12.00000                       1 5.00000
# 14.00000                       2 4.50000
# 15.00000                       2 3.00000
# 22.00000                       1 5.00000

df.head()

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

####################
# Time-Based Weighted Average

# Soru: Ne yaparsak güncel trendi ortalamaya daha iyi bir şekilde yansıtabiliriz.
# Cevap: Time - Based Weighted Average Puan zamanlarına göre ağırlıklı ortalama yaparsak eğer bu durumda örneğin son 30
# güne farklı bir ağırlık ver son 60 güne farklı bir ağırlık ver gibi kombinasyonlar ile zaman göre bir ağırlıklı
# ortalama hesaplayabiliriz.
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])#burada Timestamp değişkenin türü datetime a zaman değişkenine çevrildi

current_date = pd.to_datetime('2021-02-10 0:0:0') # bugünün tarihi belirlendi. burada parantez içersinde string
# formatında bir değer verilerek bunu zaman cinsine çevir dedik. Bu tarih bu veri setindeki maksimum tarih oldu.

df["days"] = (current_date - df["Timestamp"]).dt.days # burada bugünün tarihinden yorumun yapıldığı tarih çıkarılarak
# ve bu elde ettiğimiz yeni tarih ise days ile gün cinsine çevrilerek veri seti içerisine days isminde yeni bir değişken
# olarak atandı. Bu değişken bize yorumun en son kaç gün önce yapıldığını söylüyor.

df[df["days"] <= 30].count() # son 30 günde yapılan yorumların sayısını ekrana getirdik.

df.loc[df["days"] <= 30, "Rating"].mean() # burada son 30 günde yapılan yorumların ortalamasının ekrana getirdik.

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() # burada son 30 ile 90 gün arasında yorum yapanların
# ortalamasını ekrana getirdik.

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() # burada 90 ile 180 gün arasında yorum yapanların
# ortalamasını ekrana getirdik.

df.loc[(df["days"] > 180), "Rating"].mean() # burada yorum yapmasından 180 gün den fazla süre geçenlerin ortalamasını
# ekrana getirdik.

# Bilgi: Aslında bizim amacımız belirli farklı zaman aralıklarına farklı bir şekilde odaklanmak. Bu sebeple yukarıda
# yaptığımız belirlediğimiz zaman aralıkları için farklı farklı ağırlıklar vererek zamanın etkisini ağırlık hesabına
# yansıtabiliriz.
# Bu zamanın etkisini ağırlık hesabına yanstımak için farklı zaman aralıkları için aşağıdaki gibi ağırlıkalr belirlendi.
# Bu ağırlıklar tamamen veri setine göre kendimiz belirliyoruz kendimiz yorum yaparak karar veriyoruz.
# Biz bu veri setinde şu şekilde karar verdik:
# - 30 günden düşük yani 0 ile 30 gün içerisnde yorum yapanların etkisini %28
# - 30 ile 90 gün arasında yorum yapanların etkisini %26
# - 90 ile 180 gün arasında yorum yapanların etkisini %24
# - 180 günden fazla süre geçenlerin etkisini ise %22 olarak tayin ettik. bu oranları tamamen veri setine göre kendimiz
# belirliyoruz.
# Bu belirlediğimiz oranları aşağıdaki gibi kodumuza ekliyoruz.
# / bu ifade kodu aşağıda yazmaya devam ettiğimizi gösteriyor.

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

# yukarıda yazdığımız kodu kısaca anlatmak gerekirse:
# df["days"] <= 30, "Rating"].mean() * 28/100 kodu ile df içerisinde yer alan gün değişkenine ait 30 günden az olan
# yorumların ortalamalarının yüzde 28 i ni al dedik.
# aynı şekilde diğer kod satırlarında da yüzde 26 24 22 si uygulandı.
# BÖYLELİKLE ZAMANA GÖRE AĞIRLIKLI ORTALAMYI HESAPLAMIŞ OLDUK.

# Aşağıda ise yukarıda yaptığımız kodun fonksiyon haline getirilmiş şeklidir.

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100


time_based_weighted_average(df) # burada direk bu fonksiyon içersine dataframe i yolladğımız zaman varsayılan değerler
# ile bize zamanın etkisini dahil ederek bir ortalam verecektir.

time_based_weighted_average(df, 30, 26, 22, 22) # eğer biz varsayılan değilde farklı ağırlıklar ile hesaplama yapmak
# istersek bu şekilde farklı farklı değerler girererk hesaplama yapabiliriz.

# Biz bu çalışmada en güncel olan yorumlara daha fazla ağırlık puan vererek hesaplamış olduk.

####################
# User-Based Weighted Average (Kullanıcı Temelli Ağırlıklı Ortalama )
# User Quality
####################


# Soru: Acaba bütün kullanıcıların verdiği puanlar aynı ağırlığa mı sahip olmalı?
# Yani kursun tamamını izleyin ile kişi ile yüzde birini izleyin kişinin veridği puan aynı ağırlığa mı sahip olmalı?
# Şöylede diyebiliriz : bir kullanıcı kursu açmış kapatmış ama diğeri ise tamamını izlemiş şimdi bunların puanları aynı
# ağırlıkta mı olmalı? Referans noktamız bu olacak.
# Burada kursu izleme oranlarına göre farklı bir ağırlık yapmalıyız. Böyle bir varsayım yapıyoruz.

# Kursun ilerleme durumuna göre verilen punlar ile ilgili ağırlıklandırma yapmak istiyoruz.

####################

df.head()

df.groupby("Progress").agg({"Rating": "mean"}) # burada izleme oranına göre kırılım yapıp puanların ortalamasını aldık.

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

# yukarıda şunu yaptık kursun izlenmsi ile ağırlıklı bir şekilde puanın hesaplanmasını yaptık. bunu yaparken de
# izleme oranın %10 dan az olanların puan ortalamasının etkisinin %22 sini aldık.
# %10 ile %45 oranı arasında izleme yapanların puan ortalamasının %24 ünü aldık.
# % 45 ile %75 oranları arasında izleme yapanların puan ortalamsının %26 sını aldık.
# % 75 den fazla izleme yapanların puan ortalamasının %28 ni alarak kullanıcı tabanlı ağırlıkı ortalama hesabı yaptık.


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Weighted Rating (Ağırlıklı Derecelendirme)

# Burada time - based ve user-based hesapladığımız ağırlıklı ortalamaları burada bir araya getirirek tek bir fonlksiyon
# kullanarak yeni bir skor elde edeceğiz.
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

# yukarıda bir fonksiyon tanımladık. bu fonksiyon 3 argümana sahip
# 1- fonksiyonun içerisine girilecek olan data frame
# 2- time-based den gelen ağırlık
# 3- user-based den gelen ağırlık

# bu fonksiyonda iki ve üçüncü argümanın varsayılan değeri 50 olarak alındı.
# time based den gelen ağırlığın yüzde  50 si alındı. user based den gelen ağırlığın yüzde 50 si alındı.
# böylelikle bu fonksiyon bize return ile ağırlıklı yeni bir skor döndürecek.

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60) # eğer biz varsayılan ağırlıkları değiştişrmek istersek bu şekilde
# değiştirebiliriz.










