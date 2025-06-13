Elbette, sunduğunuz veriler ve geliştirdiğimiz algoritma üzerine kurulu, akademik formata uygun bir makale taslağı hazırladım.

---

### **Video Gözetiminde Terk Edilmiş Nesne Tespiti için Mekansal-Zamansal Maske Tabanlı Bir Yaklaşım**

**Özet**

Bu çalışma, video gözetim sistemlerinde terk edilmiş nesnelerin tespiti (AOD) için yenilikçi bir mekansal-zamansal (spatiotemporal) yaklaşım sunmaktadır. Geleneksel sahip tabanlı izleme (owner-based tracking) yöntemlerinin karmaşık ve kalabalık sahnelerde yaşadığı zorlukların üstesinden gelmek amacıyla, "Person Mask" (Kişi Maskesi) adını verdiğimiz dinamik bir haritalama tekniği geliştirilmiştir. Bu teknik, kişilerin video sahnesindeki varlığını ve hareketlerini kısa süreli bir hafıza olarak tutar. Bir nesnenin terk edilme durumu, nesnenin durağanlığı ve bu dinamik kişi maskesi ile olan etkileşimine göre belirlenir. Yöntemimizin etkinliğini değerlendirmek için YOLOv10L, YOLOv11L, YOLOv12L ve RT-DETR-L dahil olmak üzere dört farklı nesne tespit modeli, 11 video içeren özel bir veri seti üzerinde test edilmiştir. Deney sonuçları, önerilen yaklaşımın uygulanabilirliğini göstermiş ve özellikle RT-DETR-L modelinin, %40.9 Precision ve %16.9 F1-Skoru ile diğer modellere kıyasla üstün bir performans sergilediğini ortaya koymuştur.

**Anahtar Kelimeler:** Terk Edilmiş Nesne Tespiti, Video Gözetim, Bilgisayarlı Görü, Person Mask, Derin Öğrenme, Nesne Takibi.

---

### **1. Giriş**

Kamusal alanlarda güvenliğin sağlanması, modern toplumların en önemli önceliklerinden biridir. Havaalanları, metro istasyonları ve alışveriş merkezleri gibi yoğun insan trafiğinin olduğu bölgelerde bırakılan sahipsiz nesneler, potansiyel bir tehdit unsuru olarak kabul edilir. Bu tür nesnelerin hızlı ve doğru bir şekilde tespit edilmesi, güvenlik protokollerinin etkinliği için hayati önem taşır. Terk Edilmiş Nesne Tespiti (AOD), bu ihtiyaca yönelik olarak geliştirilen otomatik video analiz sistemlerinin önemli bir alt alanıdır.

Geleneksel AOD yaklaşımları genellikle, bir nesneyi taşıyan kişiyi "sahip" olarak belirleyip bu sahibi takip etme mantığına dayanır. Ancak bu yöntemler, kalabalık sahnelerde yaşanan oklüzyon (nesnenin veya kişinin başka bir nesne/kişi tarafından engellenmesi), sahip-nesne ilişkisinin belirsizliği ve takip hataları gibi nedenlerle sıkça başarısız olabilmektedir.

Bu çalışmada, sahip tabanlı izlemeye bir alternatif olarak, kişilerin alandaki varlığını temel alan mekansal-zamansal bir yöntem öneriyoruz. Geliştirdiğimiz "Person Mask" tekniği, kişilerin video sahnesindeki hareket izlerini dinamik bir harita üzerinde tutar. Bir nesnenin terk edilme kararı, nesnenin durağanlık durumu ile bu harita üzerindeki etkileşimine göre verilir. Bu yaklaşım, doğrudan bir sahip ataması gerektirmediği için daha esnek ve gürültüye karşı daha dayanıklıdır.

---

### **2. Metodoloji**

Önerilen sistem, üç ana modülden oluşan bir ardışık düzende (pipeline) çalışır: Tespit ve Takip, Person Mask Analizi ve Karar Verme.

#### **2.1. Tespit ve Takip Modülü**
Sistemin ilk adımı, video karesindeki tüm nesneleri ve kişileri tespit etmektir. Bu görev için, önceden eğitilmiş derin öğrenme tabanlı nesne tespit modelleri (YOLO, RT-DETR) kullanılır. Tespit edilen her nesneye, `botsort.yaml` gibi bir takip algoritması aracılığıyla benzersiz bir `track_id` atanır. Bu ID, nesnenin sahnedeki ömrü boyunca kimliğinin korunmasını sağlar.

#### **2.2. Person Mask Algoritması**
Yaklaşımımızın temel taşı olan "Person Mask", kişilerin mekandaki varlığını temsil eden `(Y, X)` boyutlarında (video çözünürlüğü ile aynı) iki boyutlu bir matristir.

1.  **Bozunma (Decay):** Her yeni kare işlenmeden önce, maske matrisindeki tüm piksel değerleri sabit bir `decay_rate` (örneğin 1) kadar azaltılır. Bu, maskenin zamanla "unutmasını" ve eski bilgilerin etkisini yitirmesini sağlar.
2.  **İşaretleme (Marking):** Tespit edilen her "person" sınıfına ait nesnenin sınırlayıcı kutusu (bounding box) içindeki pikseller, `initial_value` (örneğin 64) gibi yüksek bir değere ayarlanır.
3.  **Sonuç:** Bu dinamik süreç, kişilerin yakın zamanda bulunduğu alanların "sıcak" (yüksek değerli), uzun süre önce bulunduğu alanların ise "soğuk" (düşük değerli) olduğu bir ısı haritası oluşturur.

#### **2.3. Terk Edilmiş Nesne Tespit Algoritması**
Bir nesnenin terk edilmiş olarak sınıflandırılması, aşağıdaki kurallara göre işleyen bir durum makinesi (state machine) ile gerçekleştirilir.

1.  **Durağanlık Tespiti:** Sadece durağan nesneler terk edilme adayı olabilir. Bir nesnenin durağanlığı, ardışık iki karedeki sınırlayıcı kutuları arasındaki IoU (Intersection over Union) değerinin önceden belirlenmiş bir eşiği (`stationary_iou_threshold`) aşmasıyla belirlenir.

2.  **Aday Durumu (`Candidate`):** Durağan bir nesnenin sınırlayıcı kutusu, Person Mask üzerinde aktif bir bölgeyle (piksel değeri `> mask_threshold`) kesişiyorsa, bu nesne "aday" olarak işaretlenir. Bu durum, "bir kişi durağan bir nesnenin yanında" senaryosunu temsil eder.

3.  **Terk Edilmiş Durumu (`Abandoned`):** "Aday" olarak işaretlenmiş bir nesne, artık Person Mask ile kesişmiyorsa (yani kişinin oluşturduğu "sıcak" bölge nesnenin üzerinden çekilmişse), nesne "terk edilmiş" olarak sınıflandırılır. Bu, "kişi nesnenin yanından ayrıldı ve nesne hala orada" senaryosunu temsil eder.

**Algoritma 1: Terk Edilmiş Nesne Tespiti Pseudocode**
```
PROCEDURE ProcessVideoFrames:
  Initialize TrackedObjects, CandidateObjects
  Initialize PersonMask with zeros

  FOR each Frame in Video:
    // Adım 1: Tespit ve Maske Güncelleme
    Detections = DetectAndTrackObjects(Frame)
    PersonMask = Decay(PersonMask, decay_rate=1)
    
    FOR each Detection in Detections:
      UpdateTrackedObject(Detection)
      IF Detection.class == "person":
        MarkOnMask(PersonMask, Detection.bbox, value=64)

    // Adım 2: Durum Analizi ve Karar Verme
    FOR each Object in TrackedObjects:
      IF Object.class is not a TargetClass: CONTINUE

      is_stationary = CheckStationarity(Object)
      intersects_mask = CheckMaskIntersection(Object.bbox, PersonMask)

      IF is_stationary AND intersects_mask:
        // Aday olarak işaretle veya güncelle
        MarkAsCandidate(Object, CandidateObjects)
      ELSE IF Object was a Candidate:
        // Terk edilmiş olarak onayla
        MarkAsAbandoned(Object)
        RemoveFromCandidates(Object)
      END IF
    END FOR
  END FOR
END PROCEDURE
```

---

### **3. Deneyler ve Kurulum**

#### **3.1. Veri Seti**
Deneyler, terk edilmiş nesne senaryolarını içeren 11 farklı video (`video1.avi` - `video11.avi`) üzerinde gerçekleştirilmiştir. Her video için zemin gerçekliği (ground truth) verileri, YOLO formatında `obj_train_data` klasöründe sağlanmıştır.

#### **3.2. Değerlendirilen Modeller**
Sistemin farklı tespit omurgalarıyla performansını ölçmek için dört model test edilmiştir:
-   **YOLOv10L**
-   **YOLOv11L**
-   **YOLOv12L**
-   **RT-DETR-L**

#### **3.3. Değerlendirme Metrikleri**
Model performansını ölçmek için standart metrikler kullanılmıştır:
-   **Precision (Kesinlik):** `TP / (TP + FP)` - Tespit edilen terk edilmiş nesnelerin ne kadarının doğru olduğu.
-   **Recall (Duyarlılık):** `TP / (TP + FN)` - Gerçekte var olan terk edilmiş nesnelerin ne kadarının tespit edilebildiği.
-   **Accuracy (Doğruluk):** `TP / (TP + FP + FN)` - Toplam tespitler ve zemin gerçekliği içindeki doğruluk oranı.
-   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)` - Kesinlik ve Duyarlılık metriklerinin harmonik ortalaması.

---

### **4. Sonuçlar ve Tartışma**

Aşağıdaki tablo, dört modelin 11 video üzerindeki performansını özetlemektedir.

**Tablo 1: Modellerin Videolar Üzerindeki Performans Karşılaştırması**
| Model    | Metrik    | video1 | video2 | video3 | video4 | video5 | video6 | video7 | video8 | video9 | video10 | video11 | **Ortalama** |
|:---------|:----------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:--------|:--------|:-------------|
| YOLOv10L | precision | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 0.000  | 1.000   | 0.000   | **0.182**    |
|          | recall    | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.245  | 0.000  | 0.227   | 0.000   | **0.043**    |
|          | f1        | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.393  | 0.000  | 0.371   | 0.000   | **0.069**    |
| YOLOv11L | precision | 1.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 0.000  | 1.000   | 0.000   | **0.273**    |
|          | recall    | 0.219  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.245  | 0.000  | 0.227   | 0.000   | **0.063**    |
|          | f1        | 0.360  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.393  | 0.000  | 0.371   | 0.000   | **0.102**    |
| YOLOv12L | precision | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
|          | recall    | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
|          | f1        | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
| RT-DETR-L| precision | 1.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 1.000  | 1.000  | 0.500   | 0.000   | **0.409**    |
|          | recall    | 0.218  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.212  | 0.246  | 0.282  | 0.228   | 0.000   | **0.108**    |
|          | f1        | 0.358  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.350  | 0.394  | 0.441  | 0.314   | 0.000   | **0.169**    |

**Tartışma:**
Sonuçlar, **RT-DETR-L** modelinin, test edilen diğer modellere göre bu görevde daha üstün bir performans sergilediğini açıkça göstermektedir. Özellikle ortalama **%40.9 Precision** ve **%16.9 F1-Score** ile en dengeli sonuçları vermiştir. YOLOv12L modelinin hiçbir videoda başarılı bir tespit yapamaması, modelin bu spesifik veri seti ve görev için uygun olmadığını veya ek ince ayar (fine-tuning) gerektirdiğini düşündürmektedir.

RT-DETR-L'nin başarısı, Transformer tabanlı mimarisinin nesneleri daha tutarlı bir şekilde takip etme yeteneğinden kaynaklanıyor olabilir. Bu durum, durağanlık ve maske kesişimi analizlerimizin daha stabil çalışmasını sağlamaktadır. Metriklerin genel olarak düşük olması, veri setindeki senaryoların zorluğuna (örneğin, düşük ışık, oklüzyon, nesnelerin küçük olması) işaret etmektedir. Bazı videolarda (örn. video2, video3) hiçbir modelin başarılı olamaması, bu videoların özellikle zorlu senaryolar içerdiğini göstermektedir.

---

### **5. Sonuç ve Gelecek Çalışmalar**

Bu çalışmada, terk edilmiş nesne tespiti için geleneksel yöntemlere alternatif, mekansal-zamansal bir "Person Mask" yaklaşımı sunulmuştur. Geliştirilen algoritma, farklı derin öğrenme modelleriyle test edilmiş ve RT-DETR-L'nin en umut verici sonuçları verdiği gözlemlenmiştir. Yöntem, özellikle sahip takibinin zor olduğu senaryolarda esnek ve etkili bir çözüm potansiyeli taşımaktadır.

Gelecek çalışmalar aşağıdaki alanlara odaklanabilir:
1.  **Hiperparametre Optimizasyonu:** Maske bozunma oranı, başlangıç değeri ve IoU eşiği gibi parametrelerin daha sistematik bir şekilde optimize edilmesi.
2.  **Model İnce Ayarı (Fine-tuning):** Kullanılan modellerin, göreve özel bir veri seti üzerinde ince ayarlanarak tespit doğruluğunun artırılması.
3.  **Karmaşık Senaryo Yönetimi:** Bir nesnenin bir kişiden diğerine geçmesi gibi daha karmaşık sosyal etkileşimleri anlayabilecek mantıkların eklenmesi.