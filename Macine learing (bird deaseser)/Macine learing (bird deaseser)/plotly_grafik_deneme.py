#!/usr/bin/env python3
"""
interactive_plots.py
Plotly ile 5 farklı interaktif görsel
- Çizgi
- 3D yüzey
- Bar + Pie subplot
- Animasyonlu scatter
- Heatmap
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"   # veya "vscode", "notebook" vb.

# ---------- 1) İnteraktif Çizgi ----------
print("1) İnteraktif çizgi grafiği…")
x = np.linspace(0, 10, 100)
fig1 = go.Figure()
for f, color, name in [(np.sin, "blue", "sin(x)"),
                       (np.cos, "red", "cos(x)"),
                       (lambda t: np.sin(t)*np.cos(t), "green", "sin(x)×cos(x)")]:
    fig1.add_trace(go.Scatter(x=x, y=f(x), mode='lines',
                              name=name, line=dict(color=color, width=3),
                              hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'))
fig1.update_layout(title="İnteraktif Matematiksel Fonksiyonlar",
                   xaxis_title="X", yaxis_title="Y",
                   hovermode="x unified", template="plotly_white")
fig1.show()

# ---------- 2) 3D Yüzey ----------
print("2) 3D yüzey grafiği…")
x_3d = y_3d = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_3d, y_3d)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig2 = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Viridis",
                            contours={"z": {"show": True, "start": -1, "end": 1, "size": 0.2}}))
fig2.update_layout(title="3D Yüzey: z = sin(√(x² + y²))",
                   scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                   width=800, height=600)
fig2.show()

# ---------- 3) Bar + Pie (Subplot) ----------
print("3) Bar & Pie subplot…")
diller = ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust"]
kullanım = [35, 25, 15, 10, 8, 5, 2]
memnuniyet = [85, 75, 65, 70, 72, 88, 92]
colors = ['#3776ab', '#f7df1e', '#007396', '#00599c', '#239120', '#00ADD8', '#CE422B']

fig3 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Kullanım Oranları (%)", "Memnuniyet Skorları"),
                     specs=[[{"type": "bar"}, {"type": "pie"}]])

fig3.add_trace(go.Bar(x=diller, y=kullanım, text=kullanım, textposition="auto",
                      marker_color=colors,
                      hovertemplate='%{x}<br>%{y}%<extra></extra>'), row=1, col=1)
fig3.add_trace(go.Pie(labels=diller, values=memnuniyet, hole=0.3,
                      textinfo="label+percent",
                      hovertemplate='%{label}<br>%{value}%<extra></extra>'), row=1, col=2)
fig3.update_layout(title_text="Programlama Dilleri Analizi", height=500, showlegend=False)
fig3.show()

# ---------- 4) Animasyonlu Scatter ----------
print("4) Animasyonlu scatter…")
np.random.seed(42)
df_anim = pd.DataFrame({
    "Zaman": np.repeat(range(1, 6), 20),
    "X": np.random.randn(100).cumsum(),
    "Y": np.random.randn(100).cumsum(),
    "Boyut": np.random.randint(10, 50, 100),
    "Kategori": np.random.choice(["A", "B", "C"], 100)
})
fig4 = px.scatter(df_anim, x="X", y="Y", animation_frame="Zaman",
                  size="Boyut", color="Kategori", hover_name="Kategori",
                  title="Animasyonlu Veri Görselleştirme",
                  range_x=[-20, 20], range_y=[-20, 20])
fig4.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGray")))
fig4.show()

# ---------- 5) Heatmap ----------
print("5) Korelasyon heatmap…")
features = ["Özellik A", "Özellik B", "Özellik C", "Özellik D", "Özellik E"]
corr = np.random.rand(5, 5)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
fig5 = go.Figure(go.Heatmap(z=corr, x=features, y=features,
                            colorscale="RdBu", zmid=0.5,
                            text=np.round(corr, 2), texttemplate="%{text}",
                            hovertemplate="%{x} - %{y}<br>Korelasyon: %{z:.3f}<extra></extra>"))
fig5.update_layout(title="Korelasyon Matrisi Isı Haritası",
                   width=600, height=600)
fig5.show()

print("\n✅ Tüm interaktif grafikler hazır!")

