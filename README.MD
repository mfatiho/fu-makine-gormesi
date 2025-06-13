
### **Rapor: Video Akışında Maske Tabanlı Akıllı Terk Edilmiş Nesne Tespiti**

---

### **1. Özet**

Bu rapor, video gözetim sistemleri için geliştirilmiş, terk edilmiş nesneleri (abandoned objects) akıllı bir şekilde tespit eden sistemin teknik detaylarını açıklamaktadır. Proje, modern derin öğrenme tabanlı nesne tespit modellerini (YOLO, RTDETR vb.) kullanarak, "Person Mask" (Kişi Maskesi) adını verdiğimiz yenilikçi bir yöntemle nesnelerin terk edilme durumunu analiz eder.

Sistem, bir nesnenin bir kişi tarafından bırakılıp bırakılmadığını, kişinin nesne etrafındaki varlığına ve ayrılışına göre dinamik olarak belirler. Tespit edilen nesneler, durumlarına göre (normal, aday, terk edilmiş) farklı renklerde ve bilgilendirici etiketlerle görselleştirilir. Bu yaklaşım, geleneksel yöntemlere kıyasla daha esnek, doğru ve karmaşık senaryolara daha uyumlu bir çözüm sunmaktadır.

---

### **2. Giriş**

#### **2.1. Problem Tanımı**
Havaalanları, tren istasyonları, alışveriş merkezleri gibi kalabalık kamusal alanlarda sahipsiz bırakılan paketler ciddi güvenlik riskleri oluşturabilir. Bu nesnelerin manuel olarak tespiti hem verimsiz hem de insan hatasına açıktır. Otomatik terk edilmiş nesne tespiti sistemleri, bu riski minimize etmek için kritik öneme sahiptir.

**Zorluklar:**
- Bir nesnenin geçici olarak mı yoksa kalıcı olarak mı bırakıldığını ayırt etmek.
- Kalabalık sahnelerde kişi ve nesne takibinin zorluğu.
- Nesnenin sahibi olan kişiyle olan ilişkisini doğru bir şekilde kurmak.

#### **2.2. Projenin Amacı**
Bu projenin temel amacı, bir video akışında terk edilmiş nesneleri yüksek doğrulukla ve gerçek zamanlı olarak tespit eden, modüler ve yapılandırılabilir bir sistem geliştirmektir. Sistem, aşağıdaki hedeflere ulaşmayı amaçlar:
- **Doğruluk:** Yanlış alarm oranını düşürerek, sadece gerçekten terk edilmiş nesneleri işaretlemek.
- **Esneklik:** Farklı nesne tespit modelleriyle kolayca entegre olabilmek.
- **Anlaşılabilirlik:** Algoritmanın karar mekanizmasını görsel olarak kullanıcıya sunarak şeffaflık sağlamak.

#### **2.3. Kullanılan Teknolojiler**
- **Programlama Dili:** Python 3
- **Kütüphaneler:**
    - **OpenCV:** Video işleme, okuma/yazma ve görselleştirme işlemleri.
    - **Ultralytics:** YOLO, RTDETR gibi son teknoloji nesne tespiti ve takibi modellerini kullanmak için.
    - **NumPy:** Yüksek performanslı sayısal hesaplamalar, özellikle "Person Mask" operasyonları için.

---

### **3. Sistem Mimarisi ve Bileşenleri**

Sistem, "Tek Sorumluluk Prensibi" (Single Responsibility Principle) temel alınarak tasarlanmıştır. Her sınıf, belirli ve tek bir görevi yerine getirir.

- **`AppConfig` (Konfigürasyon Modülü):** Uygulamanın tüm ayarlarını barındıran merkezi bir veri sınıfıdır. FPS hedefi, IoU eşiği, maske özellikleri gibi tüm parametreler buradan yönetilir. Bu, kodda değişiklik yapmadan algoritmayı ayarlamayı sağlar.

- **`DetectionModel` (Tespit Modülü):** Derin öğrenme modelini yüklemek ve her bir video karesi üzerinde nesne tespiti/takibi yapmakla sorumludur. Farklı modeller (YOLO, RTDETR) arasında bir soyutlama katmanı görevi görür.

- **`TrackedObject` (Nesne Takip Modülü):** Tespit edilen her bir nesnenin durumunu ve geçmişini yönetir. Bir nesnenin `track_id`'si, geçmiş `bbox` ve `centroid` bilgileri, sınıf adı geçmişi (`class_names`) ve mevcut durumu (`is_abandoned`, `is_currently_stationary`) gibi kritik verileri saklar.

- **`AbandonmentDetector` (Terk Edilme Analiz Modülü):** Algoritmanın kalbidir. Tespit edilen nesneleri, "Person Mask" kullanarak analiz eder ve durumlarını (normal, aday, terk edilmiş) belirler.

- **`VideoProcessor` (Video İşleme ve Görselleştirme Modülü):** Tüm süreci yöneten orkestratör sınıftır. Videoyu okur, her kareyi `AbandonmentDetector`'a gönderir ve sonuçları (renkli kutular, etiketler, maske) ekranda görselleştirir.

---

### **4. Algoritmanın Detaylı Çalışma Prensibi**

Algoritma, birkaç ana adımdan oluşan bir ardışık düzen (pipeline) şeklinde çalışır:

#### **Adım 1: Nesne Tespiti ve Takibi**
Her video karesi, `DetectionModel` tarafından işlenir. Model, karedeki nesneleri tespit eder ve her birine benzersiz bir `track_id` atar. Bu `track_id`, nesnenin video boyunca takip edilmesini sağlar.

#### **Adım 2: "Person Mask" Oluşturma ve Güncelleme**
Bu, sistemin en yenilikçi parçasıdır. Kişilerin mekandaki varlığını kısa süreli bir "hafıza" olarak tutar.
- **Bozunma (Decay):** Her kare işlenmeden önce, maskenin tüm piksellerinin değeri `mask_decay_rate` kadar (örn. 1) azaltılır. Bu, maskenin zamanla "solmasını" veya "unutulmasını" sağlar.
- **İşaretleme (Marking):** Model tarafından "person" olarak tespit edilen her alanın `bbox`'ı alınır ve maske üzerinde bu `bbox`'a denk gelen piksellerin değeri `mask_initial_value` (örn. 128) olarak ayarlanır.
- **Sonuç:** Bu iki işlem, kişilerin yakın zamanda bulunduğu alanların daha yüksek değerli (daha "sıcak") olduğu, uzaklaştıkça ise soğuyan dinamik bir harita oluşturur.

#### **Adım 3: Durağanlık Analizi**
Bir nesnenin terk edilmiş sayılabilmesi için durağan olması gerekir.
- **İlk Hareket:** Nesne, ilk tespit edildiğinde hareket edip etmediği `check_initial_movement` ile kontrol edilir.
- **Durağanlık Tespiti:** Nesnenin birbirini takip eden iki karedeki `bbox`'ları arasındaki **Kesişim/Birleşim Oranı (IoU)** hesaplanır. Eğer IoU, `stationary_iou_threshold` (örn. 0.75) değerinden yüksekse, nesne "durağan" (`is_currently_stationary`) olarak kabul edilir.

#### **Adım 4: Aday (Candidate) Durumuna Geçiş**
Bu adım, "Bir kişi, durağan bir nesnenin yanında duruyor" senaryosunu tespit eder.
- **Koşul:** İzlenen bir nesne, **(1)** hedef sınıflardan birine aitse (örn. "backpack"), **(2)** durağan ise ve **(3)** `bbox`'ı "Person Mask" üzerinde aktif bir alanla (`> mask_threshold`) kesişiyorsa.
- **Eylem:** Bu nesne, bir `CandidateAbandonedObject` olarak işaretlenir ve `candidate_objects` listesine eklenir. Görsel olarak **sarı** bir kutu ile çerçevelenir.

#### **Adım 5: Terk Edilmiş (Abandoned) Durumuna Geçiş**
Bu adım, "Kişi, nesnenin yanından ayrıldı ve nesne hala orada" senaryosunu tespit eder.
- **Koşul:** "Aday" listesindeki bir nesnenin `bbox`'ı artık "Person Mask" ile kesişmiyorsa. Bu, kişinin nesnenin etrafındaki "sıcak" bölgeden ayrıldığı anlamına gelir.
- **Eylem:** Nesne, "aday" statüsünden "terk edilmiş" (`is_abandoned`) statüsüne yükseltilir. Aday listesinden çıkarılır ve `TrackedObject` içindeki `is_abandoned` bayrağı `True` yapılır. Görsel olarak **kırmızı** bir kutu ile çerçevelenir.

#### **Adım 6: Durumun Otomatik Sıfırlanması**
Terk edilmiş bir nesne sonsuza kadar o durumda kalamaz.
- **Koşul:** "Terk edilmiş" olarak işaretlenen bir nesne, `abandoned_max_frames` (genellikle 1 saniyeye denk gelen kare sayısı) boyunca video akışında tespit edilmezse.
- **Eylem:** `is_abandoned` bayrağı `False` konumuna getirilerek nesnenin durumu sıfırlanır. Bu, nesnenin sahneden kaldırılması durumunda yanlış alarmların devam etmesini önler.

---

### **5. Görselleştirme ve Kullanıcı Arayüzü**

Sistem, `display_video` bayrağı aktif edildiğinde, kullanıcıya zengin bir görsel geri bildirim sunar:
- **Person Mask Katmanı:** Ekranın üzerine, kişilerin bulunduğu ve yakın zamanda bulunduğu alanları gösteren yarı saydam **kırmızı** bir katman bindirilir. Bu, algoritmanın "dikkat" alanını gösterir.
- **Renkli Sınırlayıcı Kutular:**
    - **Yeşil:** Normal, aktif olarak takip edilen nesne.
    - **Sarı:** Aday terk edilmiş nesne (bir kişi yakınında).
    - **Kırmızı:** Onaylanmış terk edilmiş nesne (kişi bölgeden ayrılmış).
- **Bilgilendirici Etiketler:** Her kutunun üzerinde `ID:{id}, DURUM, {sınıf_adı}` formatında bir etiket bulunur. Bu etiket, her nesnenin kimliğini, durumunu ve sınıfını net bir şekilde belirtir.

---

### **6. Sonuç ve Gelecek Geliştirmeler**

#### **6.1. Sonuç**
Bu proje, terk edilmiş nesne tespiti problemine "Person Mask" adlı yenilikçi bir yaklaşımla çözüm sunmaktadır. Sistem, geleneksel yöntemlerin aksine, kişi ile nesne arasında doğrudan bir "sahiplik" ilişkisi kurmak yerine, mekansal ve zamansal bir ilişki kurarak daha esnek ve doğru sonuçlar üretir. Kodun modüler yapısı, gelecekteki geliştirmeler ve farklı modellerle entegrasyon için sağlam bir temel oluşturmaktadır.

#### **6.2. Gelecek Geliştirmeler**
- **Performans Optimizasyonu:** Modelin TensorRT gibi kütüphanelerle optimize edilerek daha yüksek FPS değerlerine ulaşılması.
- **Gelişmiş Uyarı Sistemi:** Tespit edilen terk edilmiş nesneler için e-posta, SMS veya diğer alarm sistemleriyle entegrasyon.
- **Karmaşık Senaryo Yönetimi:** Bir kişinin nesneyi bırakıp başka bir kişinin alması (nesne takası) gibi daha karmaşık senaryoları yönetebilecek mantık eklenmesi.
- **3D Kamera Entegrasyonu:** Derinlik sensörleri kullanarak nesne ile kişi arasındaki mesafeyi daha hassas ölçmek ve oklüzyon (nesnenin engellenmesi) durumlarını daha iyi yönetmek.
