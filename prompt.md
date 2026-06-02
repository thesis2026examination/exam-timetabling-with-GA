Daha önceden implemente ettiğin bu genetik algoritma kodu PU-Exam dataseti içindi. Bu dataset çok büyük olduğu için artık "archive" klasöründe bulunan yeni CSV dosyalarındaki (classrooms.csv, courses.csv, instructors.csv, schedule.csv, students.csv, timeslots.csv) verilerle çalışmak istiyoruz. 

Bu yeni veri kümesi üzerinde yaptığım ön analizde çok kritik iki yapısal durum tespit ettim, kodlamayı yaparken bunları mutlaka göz önünde bulundurmalısın:
1. Çakışma Yoğunluğu %100: Toplam 22 ders var ve her ders çiftinin arasında en az 40-70 arası ortak öğrenci bulunuyor. Yani zorunlu kısıtlar gereği HİÇBİR İKİ DERS aynı zaman dilimine (timeslot) atanamaz. En az 22 farklı timeslot kullanılmalıdır.
2. Sınıf Kapasitesi Yetersizliği: Ders başına ~400 öğrenci düşerken, veri kümesindeki en büyük tekil sınıf 46 kişilik, en büyük bina ise toplam 186 kapasitelidir. Bu yüzden bir dersin sınavı tek bir sınıfa veya tek bir binaya sığamaz; her ders için aynı anda 10-15 farklı sınıfı (paralel olarak) rezerve eden bir yapı kurulmalıdır.

Bu doğrultuda kodda değiştirmeni ve eklemeni istediğim noktalar şunlardır:

1. Veri Okuma ve İşleme (Parser):
- "archive" klasöründeki yeni CSV dosyalarını okuyacak; öğrenciler, dersler, hocalar, sınıflar ve mevcut öğrenci-ders ilişkilerini (schedule.csv) hafızaya alacak parser yapısını güncelle.

2. Genetik Algoritma Kromozom Gösterimi ve Uyumlaştırma:
- Arama uzayını küçük tutmak için hibrit/memetik bir yaklaşım kullanalım. Kromozom sadece "Derslerin Zaman Dilimi Sıralamasını (Timeslot)" tutsun (22 uzunluğunda bir dizi).
- Sınıf atama işini Genetik Algoritma'nın içine gömülmüş açgözlü bir sezgisel (Greedy Heuristic - Largest Capacity First veya Best-Fit Decreasing) yönteme devredelim. GA bir zaman dilimi önerdiğinde, bu sezgisel algoritma ~400 öğrenciyi sığana kadar sınıflara otomatik dağıtsın.

3. Kısıtlar (Constraints) ve Uygunluk (Fitness) Fonksiyonu:
Aşağıdaki kurallara göre ceza/ödül mekanizmasını sıfırdan kurgula:

Zorunlu Kısıtlar (Hard Constraints - Kesinlikle İhlal Edilmemeli):
- [H1] Bir öğretmen aynı anda 2 farklı sınavda görevlendirilemez (instructors.csv kontrolü).
- [H2] Aynı sınıfta aynı anda 2 farklı sınav yapılamaz (classroom_id ve timeslot_id çakışması).
- [H3] Bir öğrenci aynı anda 2 farklı sınava giremez (student_id çakışması - veri kümesinin doğası gereği aynı timeslot'a birden fazla ders atandığı an bu kısıt tetiklenmeli).
- [H4] Kapasite Kısıtı: Bir ders için atanan paralel sınıfların toplam kapasitesi, o dersi alan öğrenci sayısından küçük olamaz.

Esnek Kısıtlar (Soft Constraints - İdeal Takvim İçin Minimize Edilmeli):
- [S1] Minimum Bina Dağılımı: Veri gereği tek binaya sığamadığımız için, bir sınavın yayıldığı farklı bina sayısını (building_name) minimize et.
- [S2] Sınav Yayılımı ve Ardışıklık: Aynı öğrencinin aynı gün üst üste sınavının olması veya aynı gün birden fazla sınava girmesini engelle/cezalandır.
- [S3] Kapasite İsrafı: Atanan sınıfların toplam kapasitesi ile öğrenci sayısı arasındaki boş koltuk farkını minimize et.

4. Sezgisel Yöntemler (Heuristics):
- Başlangıç popülasyonunu tamamen rastgele üretme; En Büyük Kayıt Önceliği (Largest Enrollment First) veya Graf Boyama (Graph Coloring) mantığıyla, en çok öğrencisi olan dersleri önceliklendirerek yerleştiren bir initialization sezgisel fonksiyonu ekle.
- Mutasyon adımında rastgele değişim yerine, en çok esnek kısıt ihlaline yol açan dersi seçip onu daha iyi bir zaman/sınıf kombinasyonuna kaydıran "Sezgisel Yerel Arama Mutasyonu (Local Search Mutation)" kullan.

Mevcut kod tabanını bu mimariye göre güncelleyerek refaktör edilmiş Python kodunu paylaşır mısın?