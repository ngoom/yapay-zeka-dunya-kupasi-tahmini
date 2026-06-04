import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# ============================================
# SAYFA YAPILANDIRMASI
# ============================================
st.set_page_config(
    page_title="2026 Dünya Kupası Şampiyon Tahmini",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 2026 DÜNYA KUPASI ŞAMPİYON TAHMİN ARACI")
st.markdown("### Yapay Zeka (Random Forest) ile Şampiyon Kim Olacak? 🤖⚽")
st.markdown("---")

# ============================================
# 1. VERİYİ YÜKLE
# ============================================
url = "https://raw.githubusercontent.com/ngoom/yapay-zeka-dunya-kupasi-tahmini/refs/heads/main/worldcup_data.txt"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(url, encoding='utf-8')
        return df
    except:
        st.error("GitHub'dan veri yüklenemedi! Lütfen dosyayı manuel yükleyin.")
        uploaded_file = st.file_uploader("worldcup_data.txt dosyasını yükleyin", type=["txt", "csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            return df
        return None

with st.spinner("Veri yükleniyor..."):
    df = load_data()

if df is None:
    st.stop()

st.success(f"✅ Veri başarıyla yüklendi! {len(df)} takım bulundu.")

# ============================================
# 2. KONFEDERASYON VE SKOR DÖNÜŞÜMLERİ
# ============================================
konfederasyon_mapping = {
    'Türkiye': 'UEFA', 'Fransa': 'UEFA', 'Ispanya': 'UEFA', 'Ingiltere': 'UEFA',
    'Almanya': 'UEFA', 'Hollanda': 'UEFA', 'Portekiz': 'UEFA', 'Belçika': 'UEFA',
    'Isviçre': 'UEFA', 'Hirvatistan': 'UEFA', 'Isveç': 'UEFA',
    'Arjantin': 'CONMEBOL', 'Brezilya': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Kolombiya': 'CONMEBOL', 'Ekvador': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Fas': 'CAF', 'Senegal': 'CAF', 'Misir': 'CAF', 'Cezayir': 'CAF',
    'Tunus': 'CAF', 'Gana': 'CAF', 'Fildisi_Sahili': 'CAF',
    'Meksika': 'CONCACAF', 'ABD': 'CONCACAF', 'Kanada': 'CONCACAF', 'Panama': 'CONCACAF',
    'Japonya': 'AFC', 'Güney Kore': 'AFC', 'Iran': 'AFC', 'Suudi_Arabistan': 'AFC',
    'Avustralya': 'AFC', 'Katar': 'AFC'
}

df['Konfederasyon'] = df['Takim'].map(konfederasyon_mapping).fillna('UEFA')

konfederasyon_zorluk = {
    'UEFA': 1.00, 'CONMEBOL': 0.90, 'CAF': 0.70, 'CONCACAF': 0.55, 'AFC': 0.40       
}
df['Zorluk_Katsayisi'] = df['Konfederasyon'].map(konfederasyon_zorluk)
guclu_afrika = ['Fas', 'Senegal', 'Misir', 'Cezayir', 'Gana']
df.loc[df['Takim'].isin(guclu_afrika), 'Zorluk_Katsayisi'] = 0.90

# Başarıları puana çeviren fonksiyon
def basari_to_score(basari):
    if basari == 1: return 100
    elif basari == 2: return 80
    elif basari == 3: return 60
    elif basari == 4: return 40
    elif basari == 5: return 20
    else: return 5

# 📌 BURASI DEĞİŞTİ: Tüm geçmiş yılları puana çevirip koda dahil ediyoruz
df['Skor_2010'] = df['Basari_2010'].apply(basari_to_score)
df['Skor_2014'] = df['Basari_2014'].apply(basari_to_score)
df['Skor_2018'] = df['Basari_2018'].apply(basari_to_score)
df['Hedef_Basari'] = df['Basari_2022'].apply(basari_to_score)

# 📌 BURASI DEĞİŞTİ: Eski yılları özellik (Feature) olarak ekledik
feature_cols = ['FIFA_Sirasi', 'Son10_Galibiyet', 'Son10_Beraberlik',
                'Ortalama_Gol_At', 'Ortalama_Gol_Ye', 'Piyasa_Değeri_Milyon',
                'Kadro_Yas', 'Ev_Sahibi', 'Zorluk_Katsayisi', 
                'Skor_2010', 'Skor_2014', 'Skor_2018']

conf_dummies = pd.get_dummies(df['Konfederasyon'], prefix='Konf')
X = pd.concat([df[feature_cols], conf_dummies], axis=1)
y = df['Hedef_Basari']

# ============================================
# 3. RANDOM FOREST TAHMİNİ
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
df['RF_Guc_Skoru'] = rf.predict(X).round(0).clip(0, 100)

# SIDEBAR
with st.sidebar:
    st.header("📊 Model Performansı")
    st.metric("R² Skoru", f"{r2_score(y_test, y_pred):.3f}")
    st.metric("Ortalama Mutlak Hata", f"{mean_absolute_error(y_test, y_pred):.2f} puan")
    st.markdown("---")
    st.header("🎮 Simülasyon Ayarları")
    sim_count = st.slider("Simülasyon Sayısı", 100, 5000, 1000, step=100)
    st.markdown("---")
    st.header("📈 Kriter Ağırlıkları")
    st.info("""
    - FIFA Sıralaması: %15
    - Geçmiş Turnuvalar (2010-2018): %15
    - Son 10 Maç Formu: %5
    - Gol Performansı: %8
    - Piyasa Değeri: %12
    - 2022 Başarısı (Target): %9
    - İstikrar & Kadro Yaşı: %16
    - Ev Sahibi: %10
    - Gol Farkı: %10
    """)

# TABLO VE GRAFİK
st.header("🏆 En Güçlü 15 Takım (Random Forest)")
top_teams = df[['Takim', 'RF_Guc_Skoru', 'FIFA_Sirasi', 'Basari_2022', 'Konfederasyon']].sort_values('RF_Guc_Skoru', ascending=False).head(15)
st.dataframe(top_teams, use_container_width=True)

fig = px.bar(top_teams, x='Takim', y='RF_Guc_Skoru', color='Konfederasyon',
             title='Takım Güç Skorları', labels={'RF_Guc_Skoru': 'Güç Skoru', 'Takim': 'Takım'})
st.plotly_chart(fig, use_container_width=True)

# ============================================
# 4. SIMÜLASYON MOTORU
# ============================================
team_power_rf = dict(zip(df['Takim'], df['RF_Guc_Skoru']))
team_group = dict(zip(df['Takim'], df['Grup']))

def match_winner(team1, team2):
    power1 = team_power_rf[team1]
    power2 = team_power_rf[team2]
    diff = power1 - power2
    prob1 = 1 / (1 + np.exp(-diff / 40))
    return team1 if random.random() < prob1 else team2

def simulate_tournament():
    groups = {g: [] for g in ['A','B','C','D','E','F','G','H','I','J','K','L']}
    for team in df['Takim']:
        if team in team_group:
            groups[team_group[team]].append(team)

    group_winners, group_runners_up, group_thirds = [], [], []
    for g, team_list in groups.items():
        sorted_teams = sorted(team_list, key=lambda x: team_power_rf.get(x, 0), reverse=True)
        if len(sorted_teams) >= 1: group_winners.append(sorted_teams[0])
        if len(sorted_teams) >= 2: group_runners_up.append(sorted_teams[1])
        if len(sorted_teams) >= 3: group_thirds.append(sorted_teams[2])

    best_thirds = sorted(group_thirds, key=lambda x: team_power_rf.get(x, 0), reverse=True)[:8]
    knockout_32 = group_winners + group_runners_up + best_thirds
    random.shuffle(knockout_32)

    def run_round(teams):
        next_round = []
        for i in range(0, len(teams), 2):
            if i+1 < len(teams): next_round.append(match_winner(teams[i], teams[i+1]))
        return next_round

    r16 = run_round(knockout_32)
    qf = run_round(r16)
    sf = run_round(qf)
    finalists = run_round(sf)
    return finalists[0] if finalists else "Hata"

# ============================================
# SİMÜLASYONU ÇAĞIR
# ============================================
st.header("🎲 Turnuva Simülasyonu")

if st.button("🚀 Simülasyonu Başlat", type="primary"):
    with st.spinner(f"{sim_count} simülasyon yapılıyor..."):
        results = []
        progress_bar = st.progress(0)
        for i in range(sim_count):
            champion = simulate_tournament()
            results.append(champion)
            if (i+1) % max(1, sim_count // 10) == 0:
                progress_bar.progress((i+1)/sim_count)
        
        counter = Counter(results)
        champion_stats = counter.most_common(10)
    
    st.success(f"✅ {sim_count} simülasyon tamamlandı!")
    st.subheader("🏆 Şampiyonluk İstatistikleri")
    
    champ_df = pd.DataFrame(champion_stats, columns=['Takım', 'Şampiyonluk Sayısı'])
    champ_df['Yüzde'] = (champ_df['Şampiyonluk Sayısı'] / sim_count * 100).round(1)
    
    for i, row in champ_df.iterrows():
        if i == 0:
            st.markdown(f"### 🥇 **{row['Takım']}** → {row['Yüzde']}% şampiyonluk oranı")
        else:
            st.markdown(f"{i+1}. **{row['Takım']}** → {row['Yüzde']}%")
    
    fig2 = px.bar(champ_df.head(10), x='Takım', y='Şampiyonluk Sayısı',
                  title=f'{sim_count} Simülasyon Sonucu Şampiyonluk Dağılımı', color='Takım')
    st.plotly_chart(fig2, use_container_width=True)
    
    top_team, top_count = champion_stats[0]
    st.balloons()
    st.markdown(f"""
    <div style="background-color: gold; padding: 20px; border-radius: 10px; text-align: center; color: black;">
        <h1>🏆 {top_team} 🏆</h1>
        <h3>{top_count}/{sim_count} kez şampiyon (%{(top_count/sim_count)*100:.1f})</h3>
        <p>🎉 RANDOM FOREST MODELİNE GÖRE 2026 DÜNYA KUPASI ŞAMPİYONU! 🎉</p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📋 Tüm Takımların Detaylı Güç Skorları"):
    st.dataframe(df[['Takim', 'RF_Guc_Skoru', 'FIFA_Sirasi', 'Basari_2022', 
                     'Son10_Galibiyet', 'Piyasa_Değeri_Milyon', 'Ev_Sahibi']].sort_values('RF_Guc_Skoru', ascending=False), use_container_width=True)

st.markdown("---")
st.caption("© 2026 Dünya Kupası Şampiyon Tahmin Aracı | Yapay Zeka ile Futbol Analitiği 🤖⚽")
