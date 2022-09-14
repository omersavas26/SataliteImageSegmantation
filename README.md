[TR]
# **BTK - Huawei Kodlama Maratonu (2022 - AI)** 
## **Yapay zeka kullanarak uydu görüntüsünü segmentlere ayırma projesi**

![donusum.png](https://github.com/omersavas26/SataliteImageSegmantation/raw/main/donusum.png)

# **Önemli Açıklama**
Kaggle da geliştirme yapmaya çalışırken karşılaştığım problemleri discord grubundan paylaşmıştım. Alp bey ile yaptığım ekran paylaşımında kendisine de durumu gösterdim. En büyük problem runtime ‘ı GPU ya geçirememekti. CPU üzerinde çalışırken de aynı kodun farklı notebook ‘lar da çalışmaması gibi tutarsızlıklar tespit ettim. En sonun da (1 günüme malolsa da) Alp beyin de bilgisi dahilinde Google Colab ortamında çalışmaya başladım. Bu proje Google Colab ortamı kullanılarak geliştirilmiştir.

### **Özet**

Bu proje Ömer SAVAŞ tarafından geliştirilmiştir. Genel olarak Donanım/Zaman kısıtı dolayısı ile bilinçli olarak optimum derinlikte bir ağ oluşturulup kısıtlı bir eğitim gerçekleştirilmiştir. Daha derin bir ağ ile daha uzun süre eğitim gerçekleştirilir ise doğruluk daha da artacaktır. 

Bunun yanında disk kısıtı nedeni ile model eğitilirken callback mekanizması ile eğitim ağırlıkları peryodik olarak kaydedilememiş, bunun yerine manuel bir kayıt mekanizması kurularak hep aynı dosya üzerine ağırlıklar güncellenmiştir.

"train.ipbynb" eğitim işlemlerini gerçekleştirmek için gerekli olan dosyadır. Akış içerisindeki adımları gerçekleştirerek eğitimi tamamlar. Eğitim sonucunda elde edilen ağırlıkları kullanarak istenilen "output.csv" ve "scores.txt" dosyalarını export etmek için "predict.ipynb" dosyası kullanılmalıdır.

Ayrıca tarafınızdan gönderilen dokümanlarda geçen sorular ve aşamalar bu belge içerisinde başlık başlık açıklanmıştır. Tarafınızdan istenilen içerik başlıkları ve sorular kalın olarak vurgulanmıştır.

### **Akış**

1.   Kütüphanelerin yüklenmesi ve içeri aktarılması
2.   Yardımcı fonksiyonların ve değişkenlerin tanımlanması
3.   Veri setinin okunması ve sonrası için cache yapılması
4.   Veri setinin normalizasyonu ve uygun formatlara getirilmesi
5.   Modelin tasarlanması
6.   Eğitimin gerçekleştirilmesi ve ağırlıkların kaydedilmesi
7.   Kayıp ve doğruluk verilerinin değerlendirilmesi
8.   Test setinin tahmin işlemi ve önizlenmesi

### **Proje Açıklaması**

#### **Genel**

Bu projede kullanmak için derin öğrenme modellerinden autoencode ve mask rcnn modelleri arasında kalınmış ve autoencoder seçilmiştir. Seçilme kriterleri "Hangi makine öğrenmesi modelini seçtiniz?" başlığı altında anlatılmıştır.

İlk olarak çok küçük bir model ile bir kaç epoch eğitim gerçekleştirilerek aşağıdaki resimlere ulaşılmış ve autoencoder modelinin bu sorunu çözme konuşunda başarılı olabileceği görülmüştür. Model kaynak kodu ve ilk tahmin resimleri aşağıdadır

```python
input_img = Input(shape = (size, size, 3))
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```

![autoencoderminimodelpredict.jpg](https://omersavas.com/dosya/huawei/autoencoderminimodelpredict.jpg)

Ardından kaynak 2 de belirtilen model incelenmiş ve bizim mini autoencoderimiz ve kaynaktaki modelin arasında bir derinlik seçilmiştir. Bazı katmanlar sadeleştirilerek aşağıda özeti ve diagram resmi çizilmiş olan modelde karar kılınmıştır. Bu model yine bir kaç epoch eğitimden geçirildikten sorna önceki modelden çok daha başarılı olacağı aşağıdaki resimden de anlaşılmaktadır.

![derinmodel](https://omersavas.com/dosya/huawei/model.png)

Sonuç olarak eğitim gerçekleştirilmiş ve grafikler aşağıdadır. Donanım kaynağının yetersizliği sebebiyle eğitim küçük bir veri seti ile gerçekleştirilmiştir. Bu da ilerleyen iterasyonlarda bir miktar aşırı öğrenme yaptır. Eğer daha büyük veri seti ve daha derin ağ ile eğitim gerçekleştirilir ise doğruluk performansının artacağı düşünülmektedir.

![grafik](https://omersavas.com/dosya/huawei/acc.png)

#### **Veri kümesinde hangi veri ön işleme adımlarını uyguladınız? **
Veri okuma, formatlama, normalize etme gibi işlemler read_images() fonksiyonu içerisinde gerçekleştirildi. Önce jp2 uzantılı uydu görüntüleri RGB ye çevirildi. Ardından numpy dizisine dönüştürülüp /255 e bölündü. .tif uzantılı segment dosyaları da tek kanal olarak okunduktan sonra one hot vektörlere dönüştürüldü.

#### **Öznitelikleri nasıl belirlediniz?**
Autoencoder (derin öğrenme) modeli ile eğitim gerçekleştirildiği için ekstra bir öznitelik çıkarımı yapılmadı. Yalnızca .tif uzantılı dosyaların içindeki "0" yani tanımsız dataları için sınıf listesinin ilk elemanı olarak "undefined" eklendi. Eğitim gerçekleştirkten sonra output.csv ve socres.txt dosyaları export edilirken de undefined sınıfı görmezden gelindi.

#### **Veri kümesini modeli hazırlamak için nasıl kullandınız?**
Veri kümesi incelendiğinde Mask RCNN modeli için .tif dosyalarından, sınıfların alanlarını poligonlar olarak eğitim öncesinde hazırlanması gerektiği görüldü. Bu sebeple o modelden vazgeçilerek elimizdeki veri setine ve probleme daha uygun olacağı düşünülen autoencoder modeli kullanılmaya karar verildi.

#### **Hangi makine öğrenmesi modelini seçtiniz?**
Model olarak, görüntü işleme konusunda kendisini kanıtlamış olan derin öğrenme seçildi. Derin öğrenme yaklaşımları içerisinde de Mask RCNN ve Autoencoder üzerinde duruldu. Aslında ilk bakışta mask RCNN modelinin daha yüksek doğruluk vereceği düşünülmüş olsa da .tif uzantılı segment verisinden poligonlar halinde öznitelik konumları ayrı bir dosya olarak oluşturulması gerektiğinden[1] autoencoder yaklaşımı tercih edildi. Autoencoder modeller yapısı gereği ekstra bir öznitelik datasına ihtiyaç duymadan bilgisayarlı görü alanında kaynak bir resme bakarak istenilen resimleri üretmede gayet başarılıdır.

#### **Seçtiğiniz modelin parametrelerini nasıl belirlediniz**
Daha önce gürültü temizleme gibi çeşitli işler için autoencoder modelini kullanmıştım. Temel olarak mini bir auto encoder model ile bir kaç epoch test gerçekleştirildi ve modelin sorunun çözümüne uygun olduğuna karar verildi. 
Ardından kaynak 2 'deki model sadeleştirilerek daha performanslı çalışacak halini kullanmaya karar verildi. Sonrasında küçük bir veri seti ile bu soruna özel testler yaparak öğrenme oranı, optimizasyon algoritması gibi hiperparametreler değiştirerek daha iyi sonuçlar elde etmeye çalışıldı.

#### **Gönderilen dokümanda istenilen proje adımları hakkında açıklamalar** 
##### **1- Veri Ön-işleme**
"**Yardımcı fonksiyonların ve değişkenlerin tanımlanması**" başlığı altında "**read_images**" fonksiyonu bu işlemi gerçekleştirmektedir. Daha detaylı açıklama yine bu dokümanın "**Veri kümesinde hangi veri ön işleme adımlarını uyguladınız? **" üst başlığında bulunmaktadır.

##### **2- Makine Öğrenmesi modelinin seçimi**
Bu dokümanın "**Hangi makine öğrenmesi modelini seçtiniz?**" üst başlığında model seçimi hakkında teorik bilgi ve kıyaslamalar bulunmaktadır. Ayrıca
"**Modelin tasarlanması**" bölümünde model ve katmanları detaylı olarak kodlanmıştır. Devam eden bloklarda da modelin özeti ve diagram çizimleri vardır.

##### **3- Modelin eğitilmesi**
"**Eğitimin gerçekleştirilmesi ve ağırlıkların kaydedilmesi**" bölümünde tasarlanmış olan model eğitimi gerçekleştirilmiştir. Yukarıda da belirtildiği üzere disk kısıtı problemi sebebiyle manuel bir ağırlık ve öğrenim bilgisi kaydetme mekanizması yine aynı bölümde kodlanmıştır.

##### **4- Model performansının ölçülmesi**
Doğrulama seti ile eğitim esnasında doğruluk performansı adım adım ölçülmektedir. Ayrıca öğrenim bilgisi her adım için kaydedilmesinden dolayı kayıp ve doğruluk bilgileri eğitim kısmından sonra grafik olarak izlenmiştir. Bunun için "**Kayıp ve doğruluk verilerinin değerlendirilmesi**" bölümüne baklıabilir. Ayrıca "predict.ipynb" dosyası tahminde bulunduğu resim seti için tahmin süresi bilgisini de ölçmektedir.

##### **5- Doğrulama kümesi sonuçlarının elde edilmesi**
Bu işlem "train.ipynb" dosyasının daha karmaşıklaşmaması ve ismine uygun içeriğe sahip olması için "predict.ipynb" dosyası içerisinde gerçekleştirilmektedir. Doğrulama kümesi ve test kümesi modelin nihai ağırlıkları ile tahmin işlemine taabi tutulup, ilgili output.csv ve  scores.txt dosyaları export edilmektedir.

### **Sonuç**
Google colab üzerinde yapılan kısıtlı eğitim sonucunda bile 75% gibi bir sonuç elde edildi. Train ve Predict dosyaları oluşturuldu ve 2 dilde dokümante edildi.

### **Kişisel Değerlendirme**
Tüm kodlama boyunca temiz kod prensipleri uygulanmaya çalışılmış ve fonksiyon/değişken isimleri amacına uygun ve açıklayıcı olarak seçilmiştir. Yine de anlaşılmayan noktalar olursa aramaktan çekinmeyin (Ömer SAVAŞ: 0554 377 54 43)

Yukarıda da bahsedildiği gibi iki derin öğrenme modeli arasında kalmıştım. Mask RCCN modeli, auto encoder modeline göre kesinlikle daha hızlı çalışacaktır. Fakat daha doğru sonuç üretir mi? Denemeden bilinemez. Eğer vakit bulabilirsem .tif resimlerinden otomatik olarak poligon öznitelikleri çıkaran bir script yazarak Mask RCNN modeli ile de sonuçları karşılaştıracağım.

Model tespiti esnasında, autoencoder 'dan önce VGG16 (imagenet) modelini kullanarak transfer learning kullanmayı denedim fakat beklidiğim performans artışını göstermemesinin yanında ciddi bir performans kaybı getirdi. Yine eğer zaman bulabilirsem çeşitli denemeler yaparak daha başarılı bir model üretmeyi istiyorum. VGG16 gibi modellerin bir kısmını Auto encoderdan önce yada encoder ve decoder arasına koymayı denemek gibi opsiyonlar düşünüyorum.

Ayrıca ilk gün kaggle 'dan kaynaklanan sorun üzerinde uğraşırken motivasyon ve zaman kaybetmem dolayısıyla gözden kaçırdığım bir durum var. Yada kendimi bu bahane ile kandırıyorum :) 1000*1000 boyutunda büyük bir resim ile eğitim ve tahmin yapılması çok maliyetli. Bunun yerine resim daha küçük boyutlara parçalanarak veri setine küçük resimler şeklinde eklenmesi daha uygun olurdu.

### **Dipnotlar**
[1]: Tensorflow 'un object detection API 'ı ile RCNN modelleri transfer learning 'den de faydalanarak küçük veri setleri ile çok başarılı sonuçlar verdiği bilinmektedir. Bu sebeple ilk etapta tif resmini open cv yardımıyla Canny kenar bulma algoritmasından geçirerek alanların sınırlarını belirledim. Ardından bu sınırları kullanarak poligonları otomatik olarak oluşturmayı; oradan da API in eğitimde kullanacağı poligonlar dosyasını export etmeyi düşündüm ama zaman kısıtlı olduğu için vazgeçtim.

### **Kaynaklar**
1. https://medium.com/@omersavas26/derin-%C3%B6%C4%9Frenme-hakk%C4%B1nda-neredeyse-her-%C5%9Fey-1-91bb8ddfde0
2. https://colab.research.google.com/github/dhassault/tf-semantic-example/blob/master/01b_semantic_segmentation_basic_colab.ipynb#scrollTo=TlAIZzR600uK
3. https://colab.research.google.com/drive/1ICnxAcVKOLaDcgrHh2SvI5Rvbdmrpxsd#scrollTo=qKM9ZgMB7umJ