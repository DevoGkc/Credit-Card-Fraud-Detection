import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Uyarıları tamamen sessize alıyoruz
warnings.filterwarnings('ignore')

print("1. Adım: Gerçekçi ve Dengesiz (Imbalanced) Fraud Veri Seti Simüle Ediliyor...")
np.random.seed(42)
n_samples = 1000

# Özellikler (Features): Harcama miktarı, işlem saati, konum uygunluğu vb.
harcama_miktari = np.random.exponential(scale=100, size=n_samples)
islem_saati = np.random.randint(0, 24, size=n_samples)
guvenli_konum = np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95]) # 1 = Ev/Sık kullanılan konum, 0 = Şüpheli konum

# Fraud etiketini üretelim (Gerçek hayattaki gibi çok az olmalı: %3 fraud oranı)
# Şüpheli konumda yapılan yüksek harcamalar fraud ihtimalini artırsın
fraud_olasiligi = (harcama_miktari * 0.002) + ((1 - guvenli_konum) * 0.4)
fraud_label = np.where(fraud_olasiligi > np.percentile(fraud_olasiligi, 97), 1, 0)

df_fraud = pd.DataFrame({
    'Harcama_Miktari_TL': np.round(harcama_miktari, 2),
    'Islem_Saati': islem_saati,
    'Guvenli_Konum_Mu': guvenli_konum,
    'Fraud_Mu': fraud_label
})

print(f"Toplam İşlem Sayısı: {len(df_fraud)}")
print(f"Normal İşlem Sayısı (0): {df_fraud['Fraud_Mu'].value_counts()[0]}")
print(f"Dolandırıcılık Sayısı (1): {df_fraud['Fraud_Mu'].value_counts()[1]} (Verinin %3'ü)")

print("\n2. Adım: Veri Bilimi Modeli Eğitiliyor (Random Forest Classifier)...")
X = df_fraud[['Harcama_Miktari_TL', 'Islem_Saati', 'Guvenli_Konum_Mu']]
y = df_fraud['Fraud_Mu']

# Veriyi %75 Eğitim, %25 Test olarak bölelim (Stratify kullanarak sınıfları dengeli dağıtıyoruz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Güçlü bir toplu öğrenme (Ensemble) modeli olan Random Forest kullanıyoruz
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahminleri yapalım
y_pred = model.predict(X_test)

print("\n--- Veri Bilimi Model Değerlendirme Raporu ---")
print(classification_report(y_test, y_pred))

print("\n3. Adım: Hata Matrisi (Confusion Matrix) Görselleştiriliyor...")
# Modelin neyi doğru neyi yanlış bildiğini gösteren matris
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title('Fraud Modeli Hata Matrisi (Confusion Matrix)')
plt.ylabel('Gercek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')

# Grafiği kaydet
plt.savefig('fraud_tespiti_hata_matrisi.png', dpi=300, bbox_inches='tight')
print("\n[BAŞARILI] Grafik 'fraud_tespiti_hata_matrisi.png' adıyla klasörünüze kaydedildi!")