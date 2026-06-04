import pandas as pd
import numpy as np
import random
from collections import Counter
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================
# 1. VERİYİ YÜKLE
# ============================================

# CSV dosyasını oku (kendi dosya yolunuzu yazın)
dosya_yolu = "worldcup_data.csv"  # veya "C:/.../worldcup_data.csv"
df = pd.read_csv(dosya_yolu, encoding='utf-8')

print("=" * 70)
print("VERİ BAŞARIYLA OKUNDU!")
print("=" * 70)
print(f"Toplam {len(df)} takım var\n")

# ============================================
# 2. KONFEDERASYON VE ZORLUK KATSAYISI EKLE
# ============================================

# Konfederasyonları elle tanımla (CSV'de yoksa)
konfederasyon_mapping = {
    'Türkiye': 'UEFA', 'Fransa': 'UEFA', 'İspanya': 'UEFA', 'İngiltere': 'UEFA',
    'Almanya': 'UEFA', 'Hollanda': 'UEFA', 'Portekiz': 'UEFA', 'Belçika': 'UEFA',
    'İsviçre': 'UEFA', 'Hırvatistan': 'UEFA', 'İsveç': 'UEFA', 'Danimarka': 'UEFA',
    'Polonya': 'UEFA', 'Çekya': 'UEFA', 'İskoçya': 'UEFA', 'Avusturya': 'UEFA',
    'Norveç': 'UEFA', 'Bosna Hersek': 'UEFA', 'Rusya': 'UEFA', 'Ukrayna': 'UEFA',
    'Sırbistan': 'UEFA', 'Galler': 'UEFA', 'Finlandiya': 'UEFA', 'Slovakya': 'UEFA',
    'Arjantin': 'CONMEBOL', 'Brezilya': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Kolombiya': 'CONMEBOL', 'Ekvador': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Şili': 'CONMEBOL',
    'Fas': 'CAF', 'Senegal': 'CAF', 'Mısır': 'CAF', 'Cezayir': 'CAF',
    'Tunus': 'CAF', 'Kamerun': 'CAF', 'Gana': 'CAF', 'Fildişi Sahili': 'CAF',
    'Kongo DC': 'CAF', 'Güney Afrika': 'CAF', 'Yeşil Burun': 'CAF', 'Nijerya': 'CAF',
    'Meksika': 'CONCACAF', 'ABD': 'CONCACAF', 'Kanada': 'CONCACAF', 'Panama': 'CONCACAF',
    'Haiti': 'CONCACAF', 'Curaçao': 'CONCACAF', 'Kosta Rika': 'CONCACAF',
    'Japonya': 'AFC', 'Güney Kore': 'AFC', 'İran': 'AFC', 'Suudi Arabistan': 'AFC',
    'Avustralya': 'AFC', 'Katar': 'AFC', 'Irak': 'AFC', 'Özbekistan': 'AFC', 'Ürdün': 'AFC'
}

df['Konfederasyon'] = df['Takim'].map(konfederasyon_mapping)

# Zorluk katsayılarını tanımla
konfederasyon_zorluk = {
    'UEFA': 1.0,
    'CONMEBOL': 0.98,
    'CAF': 0.85,
    'CONCACAF': 0.70,
    'AFC': 0.55
}
df['Zorluk_Katsayisi'] = df['Konfederasyon'].map(konfederasyon_zorluk).fillna(0.70)

# Güçlü Afrika takımları için katsayıyı artır
guclu_afrika = ['Fas', 'Senegal', 'Mısır', 'Cezayir', 'Fildişi Sahili', 'Gana', 'Kongo DC', 'Tunus']
df.loc[df['Takim'].isin(guclu_afrika), 'Zorluk_Katsayisi'] = 0.90

print("KONFEDERASYON BİLGİSİ EKLENDİ ✓")
print(f"UEFA: {len(df[df['Konfederasyon']=='UEFA'])} takım")
print(f"CONMEBOL: {len(df[df['Konfederasyon']=='CONMEBOL'])} takım")
print(f"CAF: {len(df[df['Konfederasyon']=='CAF'])} takım")
print(f"CONCACAF: {len(df[df['Konfederasyon']=='CONCACAF'])} takım")
print(f"AFC: {len(df[df['Konfederasyon']=='AFC'])} takım\n")

# ============================================
# 3. RANDOM FOREST MODELİ
# ============================================

def basari_to_score(basari):
    if basari == 1: return 100   # Şampiyon
    elif basari == 2: return 80  # Final
    elif basari == 3: return 60  # Yarı final
    elif basari == 4: return 40  # Çeyrek final
    elif basari == 5: return 20  # Son 16
    else: return 5               # Katılamadı

df['Hedef_Basari'] = df['Basari_2022'].apply(basari_to_score)

# Özellikler (X)
feature_cols = ['FIFA_Sirasi', 'Son10_Galibiyet', 'Son10_Beraberlik',
                'Ortalama_Gol_At', 'Ortalama_Gol_Ye', 'Piyasa_Değeri_Milyon',
                'Kadro_Yas', 'Ev_Sahibi', 'Zorluk_Katsayisi']

# Konfederasyon one-hot encoding
conf_dummies = pd.get_dummies(df['Konfederasyon'], prefix='Konf')
X = pd.concat([df[feature_cols], conf_dummies], axis=1)
y = df['Hedef_Basari']

# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_train, y_train)

# Model performansı
y_pred = rf.predict(X_test)
print("=" * 70)
print("RANDOM FOREST MODEL PERFORMANSI")
print("=" * 70)
print(f"R² skoru: {r2_score(y_test, y_pred):.3f}")
print(f"Ortalama Mutlak Hata: {mean_absolute_error(y_test, y_pred):.2f} puan\n")

# Takımların güç puanları
df['RF_Guc_Skoru'] = rf.predict(X).round(0).clip(0, 100)

print("📊 EN GÜÇLÜ 15 TAKIM (Random Forest Tahmini):")
print(df[['Takim', 'RF_Guc_Skoru', 'Basari_2022']].sort_values('RF_Guc_Skoru', ascending=False).head(15).to_string(index=False))
print("")

# ============================================
# 4. MAÇ SİMÜLASYONU (RF PUANLARI İLE)
# ============================================

team_power_rf = dict(zip(df['Takim'], df['RF_Guc_Skoru']))
team_group = dict(zip(df['Takim'], df['Grup']))   # ← HATA DÜZELTİLDİ! bu satır eklendi

def match_winner_rf(team1, team2):
    power1 = team_power_rf[team1]
    power2 = team_power_rf[team2]
    diff = power1 - power2
    prob1 = 1 / (1 + np.exp(-diff / 15))
    return team1 if random.random() < prob1 else team2

def simulate_tournament_rf():
    groups = {g: [] for g in ['A','B','C','D','E','F','G','H','I','J','K','L']}
    for team in df['Takim']:
        groups[team_group[team]].append(team)

    group_winners = []
    group_runners_up = []
    for g, team_list in groups.items():
        if len(team_list) >= 2:
            sorted_teams = sorted(team_list, key=lambda x: team_power_rf[x], reverse=True)
            group_winners.append(sorted_teams[0])
            group_runners_up.append(sorted_teams[1])

    knockout_teams = group_winners + group_runners_up
    random.shuffle(knockout_teams)

    round_16 = []
    for i in range(0, len(knockout_teams)-1, 2):
        if i+1 < len(knockout_teams):
            round_16.append(match_winner_rf(knockout_teams[i], knockout_teams[i+1]))

    quarter_final = []
    for i in range(0, len(round_16)-1, 2):
        if i+1 < len(round_16):
            quarter_final.append(match_winner_rf(round_16[i], round_16[i+1]))

    semi_final = []
    for i in range(0, len(quarter_final)-1, 2):
        if i+1 < len(quarter_final):
            semi_final.append(match_winner_rf(quarter_final[i], quarter_final[i+1]))

    if len(semi_final) < 2:
        return semi_final[0] if semi_final else "Hata"

    finalists = []
    for i in range(0, len(semi_final)-1, 2):
        if i+1 < len(semi_final):
            finalists.append(match_winner_rf(semi_final[i], semi_final[i+1]))

    if len(finalists) >= 2:
        return match_winner_rf(finalists[0], finalists[1])
    else:
        return finalists[0] if finalists else "Hata"

# ============================================
# 5. 1000 SİMÜLASYON
# ============================================

print("=" * 70)
print("1000 SİMÜLASYON (Random Forest Güç Puanları ile)")
print("=" * 70)

results_rf = []
for i in range(1000):
    champion = simulate_tournament_rf()
    results_rf.append(champion)
    if (i+1) % 200 == 0:
        print(f"{i+1} simülasyon tamamlandı")

counter_rf = Counter(results_rf)
champion_stats_rf = counter_rf.most_common(15)

print("\n" + "=" * 70)
print("🏆 2026 DÜNYA KUPASI ŞAMPİYONLUK TAHMİNİ (Random Forest Bazlı)")
print("=" * 70)

for i, (team, count) in enumerate(champion_stats_rf, 1):
    percentage = (count / 1000) * 100
    bar = "█" * int(percentage / 2)
    print(f"{i:2}. {team:18} {count:3} kez (%{percentage:5.1f}) {bar}")

top_team_rf, top_count_rf = champion_stats_rf[0]
print("\n" + "=" * 70)
print(f"🏆 RANDOM FOREST TAHMİNİNE GÖRE 2026 ŞAMPİYONU: {top_team_rf} ({top_count_rf} kez / %{(top_count_rf/1000)*100:.1f})")
print("=" * 70)