# Model_Karsilastirma
# 📊 Sosyal Ağ Reklamları Üzerinde Sınıflandırma Modellerinin Karşılaştırmalı Analizi

Bu proje, `Social_Network_Ads.csv` veri seti üzerinde çeşitli denetimli öğrenme (Supervised Learning) sınıflandırma algoritmalarının uygulanmasını, performanslarının karşılaştırılmasını ve hiperparametre optimizasyonlarını içermektedir.

**Ders:** YZM0206 - Yapay Zeka ve Makine Öğrenmesi Laboratuvarı 10
**Geliştirici:** Ümmühan Nilay GÜNEY

---

## 🚀 Teknolojiler ve Kütüphaneler
* **Dil:** Python
* **Veri İşleme & Analiz:** Pandas, NumPy
* **Görselleştirme:** Matplotlib, Seaborn
* **Makine Öğrenmesi:** Scikit-Learn (Logistic Regression, K-NN, Naive Bayes, Decision Tree, Random Forest, SVM)

---

## 1. Keşifçi Veri Analizi (EDA) ve Veri Ön İşleme

Makine öğrenmesi modelleri kurulmadan önce veri setinin karakteristiği incelenmiş; Yaş (Age) ve Tahmini Maaş (EstimatedSalary) değişkenlerinin satın alma (Purchased) kararı üzerindeki etkisi görselleştirilmiştir.

### Dağılım ve Korelasyon
Müşterilerin satın alma durumlarına göre yaş ve maaş dağılımları kutu grafikleri (Boxplot) ile incelenmiş, özellikler arası ilişkiler korelasyon matrisi ile tespit edilmiştir.

> **Ön İşleme (Feature Scaling):** Uzaklık tabanlı çalışan K-NN ve gradyan inişi kullanan Lojistik Regresyon gibi modellerin, yüksek değerlikli Maaş değişkeni karşısında Yaş değişkenini ezmemesi için `StandardScaler` kullanılarak özellik ölçeklendirmesi yapılmıştır.

*(Görselleri eklemek için repo içine `images` klasörü açıp resimleri yükleyin)*
---

## 2. Modellerin Kurulması ve Hiperparametre Optimizasyonu

Modellerin varsayılan ayarlarında çalıştırılması genellikle ezberlemeye (overfitting) yol açtığı için `GridSearchCV` ve 5-Katlı Çapraz Doğrulama (5-Fold CV) kullanılarak en iyi parametreler bulunmuştur:

* **Logistic Regression:** `{'C': 10, 'penalty': 'l2'}` - *(Doğruluk: 0.88)*
* **K-NN:** `{'metric': 'minkowski', 'n_neighbors': 9, 'weights': 'uniform'}` - *(Doğruluk: 0.94)*
* **Naive Bayes:** `{'var_smoothing': 0.0811}` - *(Doğruluk: 0.90)*
* **Decision Tree:** `{'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 2}` - *(Doğruluk: 0.92)*
* **Random Forest:** `{'criterion': 'gini', 'max_depth': 3, 'n_estimators': 10}` - *(Doğruluk: 0.93)*
* **SVM:** `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}` - *(Doğruluk: 0.93)*

---

## 3. Gelişmiş Değerlendirme Metrikleri

### 3.1. Detaylı Sınıflandırma Raporları (Precision, Recall, F1-Score)
Doğruluk (Accuracy) metriğine ek olarak algoritmaların Sınıf 1 (Satın Alanlar) üzerindeki tahmin yetenekleri incelenmiştir.

```text
[K-NN] Modeli Raporu:
              precision    recall  f1-score   support
           0       0.98      0.92      0.95        63
           1       0.88      0.97      0.92        37
    accuracy                           0.94       100

[Random Forest] Modeli Raporu:
              precision    recall  f1-score   support
           0       0.97      0.92      0.94        63
           1       0.88      0.95      0.91        37
    accuracy                           0.93       100
