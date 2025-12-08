import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

def load_csv():
    candidates = [
        "poultry_labeled_12k.csv",
        "poultry_labeled_cleaned.csv",
        "poultry_labeled.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p, pd.read_csv(p)
    return None, None

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def summarize_df(df):
    info = {}
    info["n_rows"] = len(df)
    info["n_cols"] = df.shape[1]
    info["dtypes"] = df.dtypes.astype(str).value_counts().to_dict()
    missing_per_col = df.isna().sum()
    info["missing_total"] = int(missing_per_col.sum())
    info["missing_by_col"] = (
        pd.DataFrame({"missing": missing_per_col, "pct": missing_per_col / len(df) * 100})
        .sort_values("missing", ascending=False)
    )
    return info

def descriptive_stats(df, numeric_cols):
    return df[numeric_cols].describe().T

def frequency_distributions(df, categorical_cols):
    freqs = {}
    for c in categorical_cols:
        freqs[c] = df[c].value_counts(dropna=False)
    return freqs

def correlations(df, numeric_cols):
    return df[numeric_cols].corr(method="pearson")

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper

def build_report(df, csv_path):
    out_dir = "reports"
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    df["width"] = coerce_numeric(df.get("width"))
    df["height"] = coerce_numeric(df.get("height"))
    df["magnification"] = coerce_numeric(df.get("magnification"))
    df["image_path"] = df.get("image_path")
    df["disease"] = df.get("disease")
    df["tissue"] = df.get("tissue")
    df["source"] = df.get("source")

    numeric_cols = [c for c in ["width", "height", "magnification"] if c in df.columns]
    categorical_cols = [c for c in ["disease", "tissue", "source", "filename"] if c in df.columns]

    summary = summarize_df(df)
    stats = descriptive_stats(df, numeric_cols) if numeric_cols else pd.DataFrame()
    freqs = frequency_distributions(df, categorical_cols)
    corr = correlations(df, numeric_cols) if len(numeric_cols) >= 2 else pd.DataFrame()

    outliers = {}
    for c in numeric_cols:
        mask, lower, upper = detect_outliers_iqr(df[c].dropna())
        outliers[c] = {
            "count": int(mask.sum()),
            "lower": float(lower),
            "upper": float(upper),
        }

    dup_paths = df["image_path"].duplicated().sum() if "image_path" in df.columns else 0
    exists_checks = {}
    sample_check = df["image_path"].dropna().head(200)
    if "image_path" in df.columns:
        exists = sample_check.apply(lambda p: Path(p).exists())
        exists_checks = {
            "checked": int(len(sample_check)),
            "missing": int((~exists).sum()),
        }

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    if "disease" in df.columns:
        ax = sns.countplot(x="disease", data=df, order=df["disease"].value_counts().index)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "bar_disease.png"))
        plt.close()

    if "tissue" in df.columns:
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x="tissue", data=df, order=df["tissue"].value_counts().index)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "bar_tissue.png"))
        plt.close()

    for c in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[c].dropna(), kde=True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"hist_{c}.png"))
        plt.close()

    if set(["width", "height"]).issubset(df.columns):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x="width", y="height", hue="disease" if "disease" in df.columns else None, alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "scatter_width_height.png"))
        plt.close()

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = []
    html.append(f"<h1>Poultry Dataset Raporu</h1>")
    html.append(f"<p>Kaynak CSV: {csv_path}</p>")
    html.append(f"<p>Oluşturulma: {now}</p>")
    html.append("<h2>Genel Özellikler</h2>")
    html.append(f"<p>Toplam kayıt: {summary['n_rows']:,}</p>")
    html.append(f"<p>Sütun sayısı: {summary['n_cols']}</p>")
    html.append("<h3>Veri Tipleri Dağılımı</h3>")
    html.append("<ul>")
    for k, v in summary["dtypes"].items():
        html.append(f"<li>{k}: {v}</li>")
    html.append("</ul>")
    html.append("<h3>Eksik Veri Analizi</h3>")
    html.append(f"<p>Toplam eksik hücre: {summary['missing_total']:,}</p>")
    html.append(summary["missing_by_col"].to_html())
    html.append("<h2>İstatistiksel Analizler</h2>")
    if not stats.empty:
        html.append("<h3>Sayısal Sütunlar İçin Tanımlayıcı İstatistikler</h3>")
        html.append(stats.to_html())
    html.append("<h3>Kategorik Sütunlar İçin Frekans Dağılımları</h3>")
    for c, s in freqs.items():
        html.append(f"<h4>{c}</h4>")
        html.append(s.to_frame("count").to_html())
    if not corr.empty:
        html.append("<h3>Korelasyon Analizleri</h3>")
        html.append(corr.to_html())
    html.append("<h2>Görselleştirmeler</h2>")
    if os.path.exists(os.path.join(fig_dir, "bar_disease.png")):
        html.append("<h3>Kategorik: Disease</h3>")
        html.append(f"<img src='figures/bar_disease.png' style='max-width:800px'>")
    if os.path.exists(os.path.join(fig_dir, "bar_tissue.png")):
        html.append("<h3>Kategorik: Tissue</h3>")
        html.append(f"<img src='figures/bar_tissue.png' style='max-width:800px'>")
    for c in numeric_cols:
        fp = os.path.join(fig_dir, f"hist_{c}.png")
        if os.path.exists(fp):
            html.append(f"<h3>Dağılım: {c}</h3>")
            html.append(f"<img src='figures/hist_{c}.png' style='max-width:800px'>")
    if os.path.exists(os.path.join(fig_dir, "scatter_width_height.png")):
        html.append("<h3>İlişki: Width vs Height</h3>")
        html.append("<img src='figures/scatter_width_height.png' style='max-width:800px'>")
    html.append("<h2>Veri Kalitesi Değerlendirmesi</h2>")
    html.append("<h3>Outlier Tespiti (IQR)</h3>")
    html.append("<ul>")
    for c, o in outliers.items():
        html.append(f"<li>{c}: {o['count']} outlier, sınırlar [{o['lower']:.2f}, {o['upper']:.2f}]</li>")
    html.append("</ul>")
    html.append("<h3>Tutarsız/Problemli Kayıtlar</h3>")
    problems = []
    if "width" in df.columns and "height" in df.columns:
        problems.append(int(((df["width"] <= 0) | (df["height"] <= 0)).sum()))
    else:
        problems.append(0)
    html.append(f"<p>Geçersiz boyut değeri: {problems[0]}</p>")
    html.append(f"<p>Yinelenen image_path sayısı: {dup_paths}</p>")
    if exists_checks:
        html.append(f"<p>Dosya mevcutluk kontrolü (örnek {exists_checks['checked']}): eksik {exists_checks['missing']}</p>")
    html.append("<h3>Veri Temizleme Önerileri</h3>")
    html.append("<ul>")
    html.append("<li>Eksik numeric değerleri uygun stratejilerle doldur veya çıkar.</li>")
    html.append("<li>Kategorik sınıfları standartlaştır ve yazım hatalarını düzelt.</li>")
    html.append("<li>Yinelenen ve geçersiz dosya yollarını kaldır veya yeniden eşle.</li>")
    html.append("<li>Boyut değerlerinde aşırı marjinalleri IQR/z-score ile ele al.</li>")
    html.append("</ul>")

    html_path = os.path.join(out_dir, "dataset_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    return {
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "html_path": html_path,
        "summary": summary,
    }

def analyze_images_on_disk():
    organized_dir = Path("organized_poultry_dataset")
    microscopy_dir = Path("poultry_microscopy")
    info = {}
    if organized_dir.exists():
        total_organized = 0
        per_class = {}
        for disease_dir in sorted(organized_dir.iterdir()):
            if disease_dir.is_dir():
                image_files = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.png")) + list(disease_dir.glob("*.tif")) + list(disease_dir.glob("*.tiff"))
                count = len(image_files)
                total_organized += count
                per_class[disease_dir.name] = count
        info["organized_total"] = total_organized
        info["organized_per_class"] = per_class
    if microscopy_dir.exists():
        image_files = list(microscopy_dir.rglob("*.jpg")) + list(microscopy_dir.rglob("*.png")) + list(microscopy_dir.rglob("*.tif")) + list(microscopy_dir.rglob("*.tiff"))
        info["microscopy_total"] = len(image_files)
    return info

def main():
    csv_path, df = load_csv()
    if csv_path is None:
        print("CSV bulunamadı")
        return
    print("Analiz başlıyor:", csv_path)
    report = build_report(df, csv_path)
    disk_info = analyze_images_on_disk()
    print("Rapor üretildi:", report["html_path"])
    if disk_info:
        print("Disk görsel özeti:", disk_info)

if __name__ == "__main__":
    main()
