# Kalp_krizi_tahmini-heart_disease_prediction
Kalp krizi tahmini-Heart disease prediction
Kalp hastalığı veri seti (`heart.csv`) üzerinde dört farklı makine öğrenmesi
modelinin karşılaştırıldığı ve kullanıcı girdisiyle risk tahmini yapmayı sağlayan bir proje.

## Özellikler
- Veri ön işleme (Label Encoding, StandardScaler)
- SMOTE ile sınıf dengesizliğinin giderilmesi
- 4 model karşılaştırması:
  - Yapay Sinir Ağı (Keras)
  - Random Forest
  - Karar Ağacı
  - Lojistik Regresyon
- Konsol üzerinden etkileşimli risk tahmini

## Kurulum
pip install numpy pandas scikit-learn imbalanced-learn tensorflow
pip install -r requirements.txt

## Çalıştırma
python main.py

## Veri Seti
Heart Failure Prediction Dataset (Kaggle)

## Uyarı
Bu proje eğitim amaçlıdır, tıbbi teşhis aracı değildir.

## Dogruluk oranları
Yapay Sinir Aği Doğruluk Orani : 89.13%
Random Forest Doğruluk Orani : 70.29%
Karar Ağaçlari Doğruluk Orani : 31.16%
Lojistik Doğruluk Orani : 85.87%