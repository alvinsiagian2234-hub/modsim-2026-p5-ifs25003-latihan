import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE 5 Lantai",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stage-card {
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #F59E0B;
    }
    .warning-box {
        background-color: #FFF7ED;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ConstructionStage:
    """
    Kelas untuk memodelkan tahapan konstruksi gedung dengan faktor risiko realistis.
    Durasi dalam satuan BULAN.
    """

    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic  = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors  = risk_factors  or {}
        self.dependencies  = dependencies  or []

    def sample_duration(self, n_simulations, resource_multiplier=1.0):
        """
        Sampling durasi tahapan (distribusi triangular) + penerapan faktor risiko.
        resource_multiplier < 1.0 berarti penambahan resource (percepatan).
        """
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )

        for risk_name, rp in self.risk_factors.items():
            if rp['type'] == 'discrete':
                # Risiko diskrit: terjadi dengan probabilitas tertentu → menambah durasi
                risk_occurs = np.random.random(n_simulations) < rp['probability']
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + rp['impact']),
                    base_duration
                )
            elif rp['type'] == 'continuous':
                # Risiko kontinu: produktivitas pekerja (faktor pembagi)
                productivity = np.random.normal(rp['mean'], rp['std'], n_simulations)
                base_duration = base_duration / np.clip(productivity, 0.4, 1.5)

        return base_duration * resource_multiplier


class MonteCarloConstructionSimulation:
    """Kelas utama untuk menjalankan simulasi Monte Carlo proyek konstruksi."""

    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config  = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self._initialize_stages()

    def _initialize_stages(self):
        for name, cfg in self.stages_config.items():
            self.stages[name] = ConstructionStage(
                name=name,
                base_params=cfg['base_params'],
                risk_factors=cfg.get('risk_factors', {}),
                dependencies=cfg.get('dependencies', [])
            )

    def run_simulation(self, resource_multipliers=None):
        """Jalankan simulasi Monte Carlo dengan dependensi antar tahapan."""
        if resource_multipliers is None:
            resource_multipliers = {}

        results     = pd.DataFrame(index=range(self.num_simulations))
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times   = pd.DataFrame(index=range(self.num_simulations))

        for stage_name, stage in self.stages.items():
            mult = resource_multipliers.get(stage_name, 1.0)
            results[stage_name] = stage.sample_duration(self.num_simulations, mult)

        for stage_name, stage in self.stages.items():
            deps = stage.dependencies
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            end_times[stage_name] = start_times[stage_name] + results[stage_name]

        results['Total_Duration'] = end_times.max(axis=1)
        for stage_name in self.stages:
            results[f'{stage_name}_Start']  = start_times[stage_name]
            results[f'{stage_name}_Finish'] = end_times[stage_name]

        self.simulation_results = results
        return results

    def calculate_critical_path_probability(self):
        if self.simulation_results is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        total_duration = self.simulation_results['Total_Duration']
        records = {}
        for stage_name in self.stages:
            finish    = self.simulation_results[f'{stage_name}_Finish']
            corr      = self.simulation_results[stage_name].corr(total_duration)
            is_crit   = (finish + 0.05) >= total_duration
            records[stage_name] = {
                'probability':    np.mean(is_crit),
                'correlation':    corr,
                'avg_duration':   self.simulation_results[stage_name].mean()
            }
        return pd.DataFrame(records).T

    def analyze_risk_contribution(self):
        if self.simulation_results is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        total_var = self.simulation_results['Total_Duration'].var()
        records = {}
        for stage_name in self.stages:
            covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            records[stage_name] = {
                'variance':             self.simulation_results[stage_name].var(),
                'std_dev':              self.simulation_results[stage_name].std(),
                'contribution_percent': (covar / total_var) * 100
            }
        return pd.DataFrame(records).T


# ============================================================================
# 3. VISUALISASI
# ============================================================================
def create_distribution_plot(results):
    total = results['Total_Duration']
    mean_val   = total.mean()
    median_val = np.median(total)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total, nbinsx=50, name='Distribusi Durasi',
        marker_color='steelblue', opacity=0.75, histnorm='probability density'
    ))
    fig.add_vline(x=mean_val,   line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.1f} bln")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_val:.1f} bln")

    ci_80 = np.percentile(total, [10, 90])
    ci_95 = np.percentile(total, [2.5, 97.5])
    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="yellow",  opacity=0.2,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="orange",  opacity=0.1,
                  annotation_text="95% CI", line_width=0)

    fig.update_layout(
        title='Distribusi Durasi Total Proyek Pembangunan Gedung FITE',
        xaxis_title='Durasi Total (Bulan)',
        yaxis_title='Densitas Probabilitas',
        height=500
    )
    stats = {
        'mean': mean_val, 'median': median_val,
        'std':  total.std(), 'min': total.min(), 'max': total.max(),
        'ci_80': ci_80, 'ci_95': ci_95
    }
    return fig, stats


def create_completion_probability_plot(results):
    deadlines = np.arange(10, 31, 1)   # 10 – 30 bulan
    probs = [np.mean(results['Total_Duration'] <= d) for d in deadlines]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deadlines, y=probs, mode='lines',
        line=dict(color='darkblue', width=3),
        fill='tozeroy', fillcolor='rgba(173,216,230,0.3)',
        name='P(selesai)'
    ))

    for level, color in [(0.5,'red'), (0.8,'green'), (0.95,'blue')]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=f"{int(level*100)}%",
                      annotation_position="right")

    # Tandai 3 skenario deadline utama
    key_dl = [16, 20, 24]
    for dl in key_dl:
        idx = np.where(deadlines == dl)[0]
        if len(idx):
            p = probs[idx[0]]
            fig.add_trace(go.Scatter(
                x=[dl], y=[p], mode='markers+text',
                marker=dict(size=12, color='crimson'),
                text=[f'{p:.1%}'], textposition="top center",
                showlegend=False
            ))

    fig.add_vrect(x0=16, x1=24, fillcolor="orange", opacity=0.08,
                  annotation_text="Skenario Deadline", line_width=0)

    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        height=500
    )
    return fig


def create_critical_path_plot(critical_df):
    df = critical_df.sort_values('probability', ascending=True)
    colors = ['#DC2626' if p > 0.7 else '#F87171' for p in df['probability']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[s.replace('_', ' ') for s in df.index],
        x=df['probability'],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in df['probability']],
        textposition='auto'
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange")
    fig.update_layout(
        title='Probabilitas Critical Path per Tahapan',
        xaxis_title='Probabilitas Menjadi Critical Path',
        xaxis_range=[0, 1.0], height=500
    )
    return fig


def create_stage_boxplot(results, stages):
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    for i, stage in enumerate(stages.keys()):
        fig.add_trace(go.Box(
            y=results[stage],
            name=stage.replace('_', '\n'),
            boxmean='sd',
            marker_color=colors[i % len(colors)],
            boxpoints='outliers'
        ))
    fig.update_layout(
        title='Distribusi Durasi per Tahapan Konstruksi',
        yaxis_title='Durasi (Bulan)', height=500, showlegend=False
    )
    return fig


def create_risk_contribution_plot(risk_df):
    df = risk_df.sort_values('contribution_percent', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[n.replace('_', '\n') for n in df.index],
        y=df['contribution_percent'],
        marker_color=px.colors.qualitative.Set3,
        text=[f'{v:.1f}%' for v in df['contribution_percent']],
        textposition='auto'
    ))
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan terhadap Variabilitas Total',
        yaxis_title='Kontribusi (%)', height=400
    )
    return fig


def create_correlation_heatmap(results, stages):
    corr = results[list(stages.keys())].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[n.replace('_', '\n') for n in corr.columns],
        y=[n.replace('_', '\n') for n in corr.index],
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}', textfont={"size": 9}
    ))
    fig.update_layout(title='Matriks Korelasi Antar Tahapan', height=500)
    return fig


def create_resource_comparison_plot(base_stats, resource_stats):
    """Perbandingan durasi baseline vs skenario penambahan resource."""
    categories = ['Mean', 'P50', 'P80', 'P95']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline (Tanpa Tambahan Resource)',
        x=categories,
        y=[base_stats['mean'], base_stats['median'],
           base_stats['ci_80'][1], base_stats['ci_95'][1]],
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        name='Dengan Tambahan Resource',
        x=categories,
        y=[resource_stats['mean'], resource_stats['median'],
           resource_stats['ci_80'][1], resource_stats['ci_95'][1]],
        marker_color='seagreen'
    ))
    fig.update_layout(
        barmode='group',
        title='Perbandingan Durasi: Baseline vs Tambahan Resource',
        yaxis_title='Durasi (Bulan)', height=420
    )
    return fig


# ============================================================================
# 4. KONFIGURASI DEFAULT STUDI KASUS
# ============================================================================
DEFAULT_CONFIG = {
    "Persiapan_dan_Perizinan": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "proses_birokrasi": {
                "type": "discrete", "probability": 0.35, "impact": 0.40
            }
        }
    },
    "Pondasi_dan_Struktur_Bawah": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "kondisi_tanah_buruk": {
                "type": "discrete", "probability": 0.25, "impact": 0.30
            },
            "produktivitas_pekerja": {
                "type": "continuous", "mean": 1.0, "std": 0.15
            }
        },
        "dependencies": ["Persiapan_dan_Perizinan"]
    },
    "Struktur_Beton_5_Lantai": {
        "base_params": {"optimistic": 5, "most_likely": 7, "pessimistic": 11},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete", "probability": 0.40, "impact": 0.20
            },
            "keterlambatan_material_beton": {
                "type": "discrete", "probability": 0.30, "impact": 0.15
            },
            "produktivitas_pekerja": {
                "type": "continuous", "mean": 1.0, "std": 0.20
            }
        },
        "dependencies": ["Pondasi_dan_Struktur_Bawah"]
    },
    "Mekanikal_Elektrikal_Plumbing": {
        "base_params": {"optimistic": 3, "most_likely": 4, "pessimistic": 7},
        "risk_factors": {
            "keterlambatan_material_teknis": {
                "type": "discrete", "probability": 0.35, "impact": 0.25
            },
            "kompleksitas_instalasi": {
                "type": "continuous", "mean": 1.0, "std": 0.20
            }
        },
        "dependencies": ["Struktur_Beton_5_Lantai"]
    },
    "Finishing_dan_Interior": {
        "base_params": {"optimistic": 3, "most_likely": 5, "pessimistic": 8},
        "risk_factors": {
            "perubahan_desain_laboratorium": {
                "type": "discrete", "probability": 0.30, "impact": 0.30
            },
            "produktivitas_pekerja": {
                "type": "continuous", "mean": 1.0, "std": 0.15
            }
        },
        "dependencies": ["Mekanikal_Elektrikal_Plumbing"]
    },
    "Instalasi_Lab_Khusus": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "keterlambatan_peralatan_VR_AR": {
                "type": "discrete", "probability": 0.40, "impact": 0.35
            },
            "kompleksitas_kalibrasi": {
                "type": "continuous", "mean": 1.0, "std": 0.25
            }
        },
        "dependencies": ["Finishing_dan_Interior"]
    },
    "Uji_Coba_dan_Serah_Terima": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "temuan_saat_inspeksi": {
                "type": "discrete", "probability": 0.30, "impact": 0.50
            }
        },
        "dependencies": ["Instalasi_Lab_Khusus"]
    }
}

# ============================================================================
# 5. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    st.markdown(
        '<h1 class="main-header">🏗️ Simulasi Monte Carlo<br>Pembangunan Gedung FITE 5 Lantai</h1>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    <b>Studi Kasus:</b> Proyek pembangunan Gedung FITE 5 lantai dengan fasilitas lengkap
    (ruang kelas, lab komputer, lab elektro, lab mobile, lab VR/AR, lab game, ruang dosen, toilet,
    dan ruang serbaguna). Simulasi Monte Carlo digunakan untuk memodelkan ketidakpastian durasi
    proyek akibat faktor <b>cuaca buruk, keterlambatan material teknis, perubahan desain laboratorium,
    dan produktivitas pekerja</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")

    num_simulations = st.sidebar.slider(
        "Jumlah Iterasi Simulasi:",
        min_value=1000, max_value=50000, value=20000, step=1000,
        help="Semakin banyak iterasi, semakin akurat hasilnya."
    )

    st.sidebar.markdown("### 📋 Konfigurasi Tahapan (Bulan)")
    stages_config = {}
    for stage_name, cfg in DEFAULT_CONFIG.items():
        with st.sidebar.expander(f"🔧 {stage_name.replace('_', ' ')}", expanded=False):
            opt = st.number_input("Optimistic",  1, 24, cfg['base_params']['optimistic'],  key=f"opt_{stage_name}")
            ml  = st.number_input("Most Likely", 1, 24, cfg['base_params']['most_likely'], key=f"ml_{stage_name}")
            pes = st.number_input("Pessimistic", 1, 24, cfg['base_params']['pessimistic'], key=f"pes_{stage_name}")

        stages_config[stage_name] = {
            "base_params":  {"optimistic": opt, "most_likely": ml, "pessimistic": pes},
            "risk_factors": cfg.get("risk_factors", {}),
            "dependencies": cfg.get("dependencies", [])
        }

    st.sidebar.markdown("### 🚀 Skenario Penambahan Resource")
    st.sidebar.caption("Nilai < 1.0 berarti percepatan (misal 0.8 = 20% lebih cepat)")
    resource_mults = {}
    for stage_name in stages_config:
        resource_mults[stage_name] = st.sidebar.slider(
            stage_name.replace('_', ' '),
            min_value=0.5, max_value=1.5, value=1.0, step=0.05,
            key=f"res_{stage_name}"
        )

    run_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.8rem;color:#666;">
    <b>Keterangan:</b><br>
    • Optimistic: Estimasi terbaik<br>
    • Most Likely: Estimasi realistis<br>
    • Pessimistic: Estimasi terburuk<br>
    • CI: Confidence Interval
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if 'sim_results'   not in st.session_state: st.session_state.sim_results   = None
    if 'simulator'     not in st.session_state: st.session_state.simulator     = None
    if 'base_stats'    not in st.session_state: st.session_state.base_stats    = None
    if 'res_results'   not in st.session_state: st.session_state.res_results   = None

    # ── Jalankan simulasi ────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("Menjalankan simulasi Monte Carlo... Harap tunggu..."):
            # Baseline (tanpa resource multiplier)
            sim_base = MonteCarloConstructionSimulation(stages_config, num_simulations)
            results_base = sim_base.run_simulation()

            # Dengan resource multiplier
            sim_res  = MonteCarloConstructionSimulation(stages_config, num_simulations)
            results_res  = sim_res.run_simulation(resource_mults)

            _, base_stats = create_distribution_plot(results_base)

            st.session_state.sim_results  = results_base
            st.session_state.simulator    = sim_base
            st.session_state.base_stats   = base_stats
            st.session_state.res_results  = results_res

        st.success(f"✅ Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.")

    # ── Tampilkan hasil ────────────────────────────────────────────────────
    if st.session_state.sim_results is not None:
        results   = st.session_state.sim_results
        simulator = st.session_state.simulator
        base_stats = st.session_state.base_stats
        results_res = st.session_state.res_results

        total_duration = results['Total_Duration']

        # ── Statistik Utama ──────────────────────────────────────────────────
        st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)

        mean_val   = base_stats['mean']
        median_val = base_stats['median']
        ci_80      = base_stats['ci_80']
        ci_95      = base_stats['ci_95']

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{mean_val:.1f} bln",  "Rata-rata Durasi"),
            (c2, f"{median_val:.1f} bln", "Median Durasi"),
            (c3, f"{ci_80[0]:.1f} – {ci_80[1]:.1f}", "80% Confidence Interval"),
            (c4, f"{ci_95[0]:.1f} – {ci_95[1]:.1f}", "95% Confidence Interval"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <h3>{val}</h3>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probabilitas 3 skenario deadline ────────────────────────────────
        st.markdown('<h2 class="sub-header">📅 Probabilitas Penyelesaian per Skenario Deadline</h2>',
                    unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        for col, dl, label in [
            (sc1, 16, "16 Bulan (Optimis)"),
            (sc2, 20, "20 Bulan (Realistis)"),
            (sc3, 24, "24 Bulan (Konservatif)")
        ]:
            prob = np.mean(total_duration <= dl)
            col.metric(label=label, value=f"{prob:.1%}",
                       delta=f"{1-prob:.1%} risiko terlambat",
                       delta_color="inverse")

        # ── Tab Visualisasi ──────────────────────────────────────────────────
        st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Distribusi Durasi",
            "🎯 Probabilitas Penyelesaian",
            "🔍 Analisis Tahapan",
            "⚠️ Analisis Risiko",
            "🚀 Skenario Resource"
        ])

        with tab1:
            fig_dist, _ = create_distribution_plot(results)
            st.plotly_chart(fig_dist, use_container_width=True)
            with st.expander("📋 Detail Statistik"):
                cl, cr = st.columns(2)
                with cl:
                    st.write("**Statistik Deskriptif:**")
                    st.write(f"- Rata-rata : {base_stats['mean']:.2f} bulan")
                    st.write(f"- Median    : {base_stats['median']:.2f} bulan")
                    st.write(f"- Std Dev   : {base_stats['std']:.2f} bulan")
                    st.write(f"- Minimum   : {base_stats['min']:.2f} bulan")
                    st.write(f"- Maksimum  : {base_stats['max']:.2f} bulan")
                with cr:
                    st.write("**Confidence Intervals:**")
                    st.write(f"- 80% CI : [{ci_80[0]:.2f}, {ci_80[1]:.2f}] bulan")
                    st.write(f"- 95% CI : [{ci_95[0]:.2f}, {ci_95[1]:.2f}] bulan")

        with tab2:
            fig_prob = create_completion_probability_plot(results)
            st.plotly_chart(fig_prob, use_container_width=True)
            with st.expander("📅 Detail Probabilitas per Deadline"):
                deadlines_check = [14, 16, 18, 20, 22, 24, 26]
                rows = []
                for dl in deadlines_check:
                    p = np.mean(total_duration <= dl)
                    rows.append({"Deadline (Bulan)": dl,
                                 "Probabilitas Selesai": f"{p:.1%}",
                                 "Risiko Terlambat": f"{1-p:.1%}"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab3:
            cl, cr = st.columns(2)
            with cl:
                crit = simulator.calculate_critical_path_probability()
                st.plotly_chart(create_critical_path_plot(crit), use_container_width=True)
            with cr:
                st.plotly_chart(create_stage_boxplot(results, simulator.stages), use_container_width=True)
            with st.expander("🔍 Detail Critical Path"):
                st.dataframe(crit.sort_values('probability', ascending=False), use_container_width=True)

        with tab4:
            cl, cr = st.columns(2)
            risk_contrib = simulator.analyze_risk_contribution()
            with cl:
                st.plotly_chart(create_risk_contribution_plot(risk_contrib), use_container_width=True)
            with cr:
                st.plotly_chart(create_correlation_heatmap(results, simulator.stages), use_container_width=True)
            with st.expander("📋 Detail Kontribusi Risiko"):
                st.dataframe(risk_contrib, use_container_width=True)

        with tab5:
            st.markdown("""
            <div class="info-box">
            Skenario ini membandingkan durasi proyek <b>baseline</b> dengan skenario di mana
            resource ditambahkan (pekerja khusus, alat berat, insinyur) pada tahapan tertentu.
            Atur <b>multiplier resource</b> di sidebar (nilai &lt; 1.0 = percepatan).
            </div>
            """, unsafe_allow_html=True)

            _, res_stats = create_distribution_plot(results_res)
            st.plotly_chart(
                create_resource_comparison_plot(base_stats, res_stats),
                use_container_width=True
            )

            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric("Rata-rata Baseline",      f"{base_stats['mean']:.1f} bln")
                st.metric("Rata-rata + Resource",    f"{res_stats['mean']:.1f} bln",
                           delta=f"{res_stats['mean']-base_stats['mean']:.1f} bln",
                           delta_color="inverse")
            with rc2:
                for dl, label in [(16,'16 Bln'), (20,'20 Bln'), (24,'24 Bln')]:
                    pb = np.mean(results['Total_Duration']     <= dl)
                    pr = np.mean(results_res['Total_Duration'] <= dl)
                    st.metric(
                        label=f"P(selesai ≤ {label})",
                        value=f"{pr:.1%}",
                        delta=f"{pr-pb:+.1%} vs baseline"
                    )

        # ── Statistik per Tahapan ─────────────────────────────────────────
        st.markdown('<h2 class="sub-header">📋 Statistik Durasi per Tahapan</h2>', unsafe_allow_html=True)
        rows = {}
        for stage in simulator.stages:
            d = results[stage]
            rows[stage] = {
                'Mean (bln)': round(d.mean(), 2),
                'Std Dev':    round(d.std(),  2),
                'Q1':         round(np.percentile(d, 25), 2),
                'Median':     round(np.percentile(d, 50), 2),
                'Q3':         round(np.percentile(d, 75), 2),
                'P95':        round(np.percentile(d, 95), 2),
            }
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

        # ── Rekomendasi ──────────────────────────────────────────────────
        st.markdown('<h2 class="sub-header">🎯 Analisis Deadline & Rekomendasi</h2>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            target = st.number_input("Masukkan deadline target (bulan):",
                                     min_value=10, max_value=36, value=20, step=1)
            prob_target = np.mean(total_duration <= target)
            days_at_risk = max(0, np.percentile(total_duration, 95) - target)
            st.metric(
                label=f"Probabilitas selesai ≤ {target} bulan",
                value=f"{prob_target:.1%}",
                delta=f"Potensi keterlambatan: {days_at_risk:.1f} bln" if days_at_risk else "Aman",
                delta_color="inverse"
            )
        with col_b:
            safety_buf  = np.percentile(total_duration, 80) - mean_val
            contingency = np.percentile(total_duration, 95) - mean_val
            st.markdown(f"""
            <div class="info-box">
                <h4>🏗️ Rekomendasi Manajemen Risiko:</h4>
                • <b>Safety Buffer</b> (80% confidence): <b>+{safety_buf:.1f} bulan</b><br>
                • <b>Contingency Reserve</b> (95% confidence): <b>+{contingency:.1f} bulan</b><br><br>
                • <b>Jadwal yang direkomendasikan:</b><br>
                &emsp;{mean_val:.1f} + {safety_buf:.1f} = <b>{mean_val + safety_buf:.1f} bulan</b>
            </div>
            """, unsafe_allow_html=True)

        # ── Info Teknis ───────────────────────────────────────────────────
        with st.expander("ℹ️ Informasi Teknis Simulasi"):
            st.write(f"- Jumlah iterasi : {num_simulations:,}")
            st.write(f"- Jumlah tahapan : {len(simulator.stages)}")
            st.write(f"- Distribusi durasi tahapan : Triangular (Optimistic, Most Likely, Pessimistic)")
            st.write(f"- Faktor risiko : Diskrit (probabilistik) & Kontinu (produktivitas normal)")
            st.write("**Faktor Risiko yang Dimodelkan:**")
            st.write("- 🌧️ Cuaca buruk (diskrit)")
            st.write("- 📦 Keterlambatan material teknis/khusus (diskrit)")
            st.write("- 🔄 Perubahan desain laboratorium (diskrit)")
            st.write("- 👷 Produktivitas pekerja (kontinu – distribusi normal)")
            st.write("- 🏛️ Proses birokrasi perizinan (diskrit)")

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem;background:#F8FAFC;border-radius:10px;">
            <h3>🏗️ Siap memulai simulasi?</h3>
            <p>Atur parameter di sidebar kiri, lalu klik <b>"Run Simulation"</b>.</p>
            <p>📊 Hasil akan ditampilkan di sini setelah proses selesai.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sub-header">📋 Preview Konfigurasi Tahapan</h2>', unsafe_allow_html=True)
        for stage_name, cfg in DEFAULT_CONFIG.items():
            bp = cfg['base_params']
            st.markdown(f"""
            <div class="stage-card">
            <b>{stage_name.replace('_', ' ')}</b> &nbsp;|&nbsp;
            ⚡ Optimistic: {bp['optimistic']} bln &nbsp;|&nbsp;
            📌 Most Likely: {bp['most_likely']} bln &nbsp;|&nbsp;
            ⚠️ Pessimistic: {bp['pessimistic']} bln
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;font-size:0.9rem;">
    <b>Simulasi Monte Carlo – Pembangunan Gedung FITE 5 Lantai</b><br>
    ⚠️ Hasil simulasi adalah estimasi probabilistik, bukan prediksi pasti.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
