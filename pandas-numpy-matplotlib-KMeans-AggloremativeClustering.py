import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')


#bir alt satırda veriler.iloc[:,3:] yazdığımızda values yazmadığımızda sağ pencerede X'in tipi dataframe oluyor ve sütun isimleri Hacim ve Maaş oluyor. alt satırdaki gibi values yazdığımızda sütun isimleri 0 ve 1 oluyor.

X = veriler.iloc[:,3:].values #obje olarak X'e tanımladık ve verilerin içinden tüm satırları alıp 3 ve 3ten sonraki sütunları çektik.
#sağdaki veriler'e tıklayınca hacim ve maaş'a göre bir kümeleme yapmak istiyoruz burda.


# yeni=np.array([[69900,79000,85500],[6325,5200,7825]]).T

from sklearn.cluster import KMeans #K ortalamalar algoritması

model = KMeans ( n_clusters = 3, init = "k-means++",random_state=0) #n_clusters = 3 yani 3 kümeye ayırmak istiyoruz. init= "k-means++" init de kullancağımız algoritmayı gösteriyor.algoritmamız k-means++

#n_clusters=3 diyince sağ pencerede gruplar'a tıkladığımızda 0,1,2 diye 3 gruba ayırıyor. 2 deseydik 0,1 diye ayıracaktı.


model.fit(X) #modelimizi eğitiyoruz.


gruplar=model.predict(X)

#x=X[:,0] #x X'in ilk sütunu. [:,0] demek tüm satırları ve 0.sütunu çekti.
#y=X[:,1] #y X'in ikinci sütunu. [:,1] demek tüm satırları ve 1.sütunu çekti.
#plt.scatter(x,y) #x ve y'yi çizdirdik.


merkezler=model.cluster_centers_ #merkezler diye bir değişken tanımladık. model.cluster_centers_ yani kümelerin merkezlerini gösteriyor. yani kümelerin merkezlerini çizdirmek istiyoruz.

x=X[:,0]
y=X[:,1]


merkez_x=merkezler[:,0]
merkez_y=merkezler[:,1]

#burda x'lerin merkezini 0.sütun olarak, y'lerin merkezini 1.sütun olarak çektik.


plt.scatter(x,y, c=gruplar) #ilk önce x ve y'yi çizdiriyoruz. yani tahmin ettiğimiz X verisinin x ve y'sine göre çizdiriyoruz.
#c=gruplar c color parametresi. böylece hangi noktaya hangi grup verdiyse mesela 0.grupları bir renkte çizecek. 1.grupları bir renkte çizecek vs vs. grup renklerini ayıracak.


plt.scatter(merkez_x,merkez_y,marker="P", s=100, c="r") # merkez_x,merkez_y çizdircez. "P" marker'ı + olarak çizdiriyor. c="r" merkezlerin renklerini görelim diye kırmızı renkte çizdircez.

#burda çalıştırınca plots'a gelince grupları renklerle ayırmış ve merkezlerini çizdirmiş olduk.


#plt.xlabel('Hacim')
#plt.ylabel('Maaş')
#plt.title('Kümelenmiş Veiler Grafiği')


gruplardt=pd.DataFrame(gruplar,columns=["Grup Numarası"]) #burda değişken atıyoruz gruplardt diye. Dataframe oluşturup girdiler olarak gruplar. sütunlar da grup numarası diye.

grup1=gruplardt[(gruplardt["Grup Numarası"]==0)] 
grup2=gruplardt[(gruplardt["Grup Numarası"]==1)]
grup3=gruplardt[(gruplardt["Grup Numarası"]==2)]

#bu üstte filtreleme işi yapıyoruz. grup numarası 0 olanları grup1 adı altında topla. grup numarası 1 olanları grup2 adı altında topla. grup numarası 2 olanları da grup3 adı altında topla diyoruz.


plt.figure() #bunu yazmamızın sebebi aynı grafik üstünde çizdirmemek, ayrı ayrı grafikler çizdirmek.
sonuclar = [] #sonuclar isminde boş bir liste tanımlıyoruz.


#döngü kuruyoruz
for i in range(1,11): #1 den 11e kadar 11 dahil değil.
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123) #kmeans tanımlıyoruz. n_clusters=i i'yi 1 alcak bir kümeye ayıracak, i'yi 2 alcak bir kümeye ayıracak vs vs.
    kmeans.fit(X) #modeli eğitiyoruz.
    sonuclar.append(kmeans.inertia_) #sonuclar.append listeye eleman eklemek için kullandık. kmeans.inertia_ yani merkezlere olan uzaklıklar toplamını hesaplayan yöntemdir.

#burda i yi 1 aldığında uzaklıkları hesaplayıp sonuclara eklicek. i yi 2 alcak uzaklıkları hesaplayıp sonuclara eklicek. diye devam ediyor i'yi 10 alana kadar.

    
plt.plot(range(1,11),sonuclar) #x'ler 1den 11e kadar 11 dahil değil. x'i 1 alıp sonuclar listesindeki ilk elemena karşılık gelen ilk noktayı çizcek. x'i 2 alıp sonuclar listesindeki ikinci elemana karşılık gelen noktayı çizecek. diye devam ediyor.

#plt.xlabel('k değeri')
#plt.ylabel('WCSS değeri')
#plt.title('k parametre seçimi')
#plt.figure()





#Hiyerarşik Kümeleme(Bölütleme)

#Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward')) #sch.linkage() içinde ilk olarak kümemiz hangisiyse onu yazıyoruz. yine aynı musteriler veri setiyle çalıştığımız için kümemiz X'ti o yüzden X yazdık. kümeler arası uzaklık methodumuzu da ward yazdık.
#bi üst satırda methodda ward yerine single yazınca grafik değişiyor


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward') #n_clusters küme sayımız, affinity noktalar arasındaki uzaklık, linkage kümeler arasındaki hangi uzaklığı kullanacağımız parametreleri.
tahmin = ac.fit_predict(X) #tahmin diyip modelimizi eğitip tahmin ettiriyoruz.

gruplarh=pd.DataFrame(tahmin,columns=['Grup Numarası'])

gruph1=gruplarh[(gruplarh['Grup Numarası']==0)]
gruph2=gruplarh[(gruplarh['Grup Numarası']==1)]
gruph3=gruplarh[(gruplarh['Grup Numarası']==2)]
gruph4=gruplarh[(gruplarh['Grup Numarası']==3)]

#burda da filtreleme işi yaptık. grup numarası 0 olanları gruph1'e, grup numarası 1 olanları gruph2'ye... diye devam ediyor.

plt.figure()

plt.scatter(X[tahmin==0,0],X[tahmin==0,1],s=40, c='red')
plt.scatter(X[tahmin==1,0],X[tahmin==1,1],s=40, c='blue')
plt.scatter(X[tahmin==2,0],X[tahmin==2,1],s=40, c='green')
plt.scatter(X[tahmin==3,0],X[tahmin==3,1],s=40, c='yellow')
plt.title('Hiyerarşik Bölütleme')
plt.show()






