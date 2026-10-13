import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

df = pd.read_csv('heart.csv')

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

X = df.drop('HeartDisease' , axis=1)
y = df['HeartDisease']


X_train, X_test , y_train, y_text = train_test_split(X,y, test_size=0.3 , random_state=42)


smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


print('Model 1: Yapay Sinir Ağlari')
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1] ,activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy' , optimizer = optimizer , metrics=['accuracy'])

model.fit(X_train_scaled, y_train_smote, epochs=100 , verbose=1 , validation_data=(X_test_scaled, y_text) )
loss, accuracy = model.evaluate(X_test_scaled, y_text)

print('Model 2 : Random Forest')
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_text, y_pred_rf)

print('Model 3: Karar Ağaçları')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote , y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_text , y_pred_dt)


print('Model 4: lojistijk regresyon modeli')
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_smote , y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_text, y_pred_lr)


print(f'Yapay Sinir Aği Doğruluk Orani : {accuracy * 100:.2f}%')
print(f'Random Forest Doğruluk Orani : {rf_accuracy * 100:.2f}%')
print(f'Karar Ağaçlari Doğruluk Orani : {dt_accuracy * 100:.2f}%')
print(f'Lojistik Doğruluk Orani : {lr_accuracy * 100:.2f}%')

while True:
    user_input_1 = float(input('Yaşinizi Giriniz: '))
    user_input_2 = input('Cinsiyetinizi Giriniz (M, F): ')
    user_input_3 = input('Göğüs Ağrisi Tipini Giriniz (ATA, NAP, ASY, TA): ')
    user_input_4 = float(input('Dinlenme Kan Basincinizi Giriniz: '))
    user_input_5 = float(input('Kolesterol Değerinizi Giriniz: '))

    try:
      
        user_data = pd.DataFrame({
            'Age': [user_input_1],
            'Sex': [label_encoder_sex.transform([user_input_2])[0]],  
            'ChestPainType': [label_encoder_cp.transform([user_input_3])[0]],  
            'RestingBP': [user_input_4],
            'Cholesterol': [user_input_5],
            'FastingBS': [0], 
            'RestingECG': [0],  
            'MaxHR': [150],  
            'ExerciseAngina': [0], 
            'Oldpeak': [0.0],
            'ST_Slope': [1]
        })


        user_data_scaled = scaler.transform(user_data)


        prediction = model.predict(user_data_scaled)
        print(f'Tahmin sonucu: Kalp hastaliği risk oraniniz: {prediction[0][0]:.4f}')

    except ValueError as e:
        print(f"Bir hata oluştu: {e}. Lütfen girdiğiniz bilgileri kontrol edin.")


    devam = input('Başka bir tahmin yapmak istiyor musunuz? (E/H): ')
    if devam.lower() != 'e':
        break