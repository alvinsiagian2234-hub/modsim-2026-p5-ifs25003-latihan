
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
    page_title="Simulasi Monte Carlo - Estimasi Waktu Proyek",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border-left: 3px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ProjectStage:
    """Kelas untuk memodelkan tahapan proyek dengan kompleksitas yang lebih realistis"""
    
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []
        
    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        """Sampling durasi dengan mempertimbangkan distribusi dan faktor risiko"""
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )
        
        total_risk_effect = 1.0
        
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                probability = risk_params['probability']
                impact = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )
                
            elif risk_params['type'] == 'continuous':
                mean = risk_params['mean']
                std = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)
        
        return base_duration * risk_multiplier

class MonteCarloProjectSimulation:
    """Kelas untuk menjalankan simulasi Monte Carlo yang lebih kompleks"""
    
    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self.initialize_stages()
        
    def initialize_stages(self):
        """Inisialisasi objek tahapan dari konfigurasi"""
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )
    
    def run_simulation(self):
        """Menjalankan simulasi Monte Carlo lengkap"""
        results = pd.DataFrame(index=range(self.num_simulations))
        
        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)
        
        results_with_deps = results.copy()
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))
        
        for stage_name in self.stages.keys():
            deps = self.stages[stage_name].dependencies
            
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            
            end_times[stage_name] = start_times[stage_name] + results[stage_name]
        
        results['Total_Duration'] = end_times.max(axis=1)
        
        for stage_name in self.stages.keys():
            results[f'{stage_name}_Finish'] = end_times[stage_name]
            results[f'{stage_name}_Start'] = start_times[stage_name]
        
        self.simulation_results = results
        return results
    
    def calculate_critical_path_probability(self):
        """Menghitung probabilitas setiap tahapan berada di critical path"""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        critical_path_probs = {}
        total_duration = self.simulation_results['Total_Duration']
        
        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            correlation = self.simulation_results[stage_name].corr(total_duration)
            delay_threshold = np.percentile(self.simulation_results[stage_name], 90)
            is_critical = (stage_finish + 0.1) >= total_duration
            prob_critical = np.mean(is_critical)
            
            critical_path_probs[stage_name] = {
                'probability': prob_critical,
                'correlation': correlation,
                'avg_duration': self.simulation_results[stage_name].mean()
            }
        
        return pd.DataFrame(critical_path_probs).T
    
    def analyze_risk_contribution(self):
        """Analisis kontribusi risiko terhadap variabilitas total durasi"""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        total_var = self.simulation_results['Total_Duration'].var()
        contributions = {}
        
        for stage_name in self.stages.keys():
            stage_var = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            contribution = (stage_covar / total_var) * 100
            
            contributions[stage_name] = {
                'variance': stage_var,
                'contribution_percent': contribution,
                'std_dev': np.sqrt(stage_var)
            }
        
        return pd.DataFrame(contributions).T

# ============================================================================
# 3. FUNGSI VISUALISASI PLOTLY
# ============================================================================
def create_distribution_plot(results):
    """Membuat plot distribusi durasi total proyek"""
    total_duration = results['Total_Duration']
    mean_duration = total_duration.mean()
    median_duration = np.median(total_duration)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=total_duration,
        nbinsx=50,
        name='Distribusi Durasi',
        marker_color='skyblue',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Garis mean dan median
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_duration:.1f} hari")
    fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_duration:.1f} hari")
    
    # Confidence intervals
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])
    
    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="yellow", opacity=0.2,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="orange", opacity=0.1,
                  annotation_text="95% CI", line_width=0)
    
    fig.update_layout(
        title='Distribusi Durasi Total Proyek',
        xaxis_title='Durasi Total Proyek (Hari)',
        yaxis_title='Densitas Probabilitas',
        showlegend=True,
        height=500
    )
    
    return fig, {
        'mean': mean_duration,
        'median': median_duration,
        'std': total_duration.std(),
        'min': total_duration.min(),
        'max': total_duration.max(),
        'ci_80': ci_80,
        'ci_95': ci_95
    }

def create_completion_probability_plot(results):
    """Membuat plot probabilitas penyelesaian proyek"""
    deadlines = np.arange(40, 101, 2)
    completion_probs = []
    
    for deadline in deadlines:
        prob = np.mean(results['Total_Duration'] <= deadline)
        completion_probs.append(prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=deadlines,
        y=completion_probs,
        mode='lines',
        name='Probabilitas Selesai',
        line=dict(color='darkblue', width=3),
        fill='tozeroy',
        fillcolor='rgba(173, 216, 230, 0.3)'
    ))
    
    # Garis referensi probabilitas
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="50%", annotation_position="right")
    fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                  annotation_text="80%", annotation_position="right")
    fig.add_hline(y=0.95, line_dash="dash", line_color="blue",
                  annotation_text="95%", annotation_position="right")
    
    # Area deadline realistis
    fig.add_vrect(x0=60, x1=80, fillcolor="orange", opacity=0.1,
                  annotation_text="Deadline Realistis", line_width=0)
    
    # Tandai deadline penting
    key_deadlines = [60, 70, 80, 90]
    for dl in key_deadlines:
        idx = np.where(deadlines == dl)[0]
        if len(idx) > 0:
            prob = completion_probs[idx[0]]
            fig.add_trace(go.Scatter(
                x=[dl], y=[prob],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=[f'{prob:.1%}'],
                textposition="top center",
                showlegend=False
            ))
    
    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Hari)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[40, 100],
        height=500
    )
    
    return fig

def create_critical_path_plot(critical_analysis):
    """Membuat plot analisis critical path"""
    critical_analysis = critical_analysis.sort_values('probability', ascending=True)
    
    fig = go.Figure()
    
    colors = ['red' if prob > 0.7 else 'lightcoral' for prob in critical_analysis['probability']]
    
    fig.add_trace(go.Bar(
        y=[stage.replace('_', ' ') for stage in critical_analysis.index],
        x=critical_analysis['probability'],
        orientation='h',
        marker_color=colors,
        text=[f'{prob:.1%}' for prob in critical_analysis['probability']],
        textposition='auto'
    ))
    
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title='Analisis Critical Path per Tahapan',
        xaxis_title='Probabilitas Menjadi Critical Path',
        xaxis_range=[0, 1.0],
        height=500
    )
    
    return fig

def create_stage_boxplot(results, stages):
    """Membuat boxplot distribusi durasi per tahapan"""
    stage_names = list(stages.keys())
    stage_data = [results[stage] for stage in stage_names]
    
    fig = go.Figure()
    
    for i, (stage, data) in enumerate(zip(stage_names, stage_data)):
        mean_val = np.mean(data)
        
        fig.add_trace(go.Box(
            y=data,
            name=stage.replace('_', '\n'),
            boxmean='sd',
            marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title='Distribusi Durasi per Tahapan',
        yaxis_title='Durasi (Hari)',
        height=500,
        showlegend=False
    )
    
    return fig

def create_risk_contribution_plot(risk_contrib):
    """Membuat plot kontribusi risiko per tahapan"""
    risk_contrib = risk_contrib.sort_values('contribution_percent', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[name.replace('_', '\n') for name in risk_contrib.index],
        y=risk_contrib['contribution_percent'],
        marker_color=px.colors.qualitative.Set3,
        text=[f'{contrib:.1f}%' for contrib in risk_contrib['contribution_percent']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan',
        yaxis_title='Kontribusi terhadap Variabilitas (%)',
        height=400
    )
    
    return fig

def create_correlation_heatmap(results, stages):
    """Membuat heatmap korelasi antar tahapan"""
    correlation_matrix = results[list(stages.keys())].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=[name.replace('_', '\n') for name in correlation_matrix.columns],
        y=[name.replace('_', '\n') for name in correlation_matrix.index],
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Matriks Korelasi Antar Tahapan',
        height=500
    )
    
    return fig

# ============================================================================
# 4. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    # Header aplikasi
    st.markdown('<h1 class="main-header">📊 Simulasi Monte Carlo - Estimasi Waktu Proyek</h1>', unsafe_allow_html=True)
    
    # Deskripsi
    st.markdown("""
    <div class="info-box">
    Estimasi waktu penyelesaian proyek pengembangan aplikasi sering kali sulit dilakukan secara deterministik 
    karena dipengaruhi oleh berbagai faktor yang tidak pasti. Aplikasi ini menggunakan simulasi Monte Carlo 
    untuk memodelkan ketidakpastian dan menghasilkan estimasi waktu penyelesaian proyek yang lebih akurat dan informatif.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk konfigurasi
    st.sidebar.markdown('<h2>⚙️ Konfigurasi Simulasi</h2>', unsafe_allow_html=True)
    
    # Slider untuk jumlah simulasi
    num_simulations = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:',
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help='Semakin banyak iterasi, semakin akurat hasilnya tetapi lebih lama waktu prosesnya'
    )
    
    st.sidebar.markdown('<h3>📋 Konfigurasi Tahapan Proyek</h3>', unsafe_allow_html=True)
    
    # Konfigurasi default
    default_config = {
        "Analisis_Kebutuhan": {
            "base_params": {"optimistic": 4, "most_likely": 6, "pessimistic": 9},
            "risk_factors": {
                "perubahan_requirement": {
                    "type": "discrete",
                    "probability": 0.3,
                    "impact": 0.25
                }
            }
        },
        "Desain_Arsitektur": {
            "base_params": {"optimistic": 5, "most_likely": 8, "pessimistic": 12},
            "risk_factors": {
                "review_iteration": {
                    "type": "discrete",
                    "probability": 0.4,
                    "impact": 0.15
                }
            },
            "dependencies": ["Analisis_Kebutuhan"]
        },
        "Implementasi_Frontend": {
            "base_params": {"optimistic": 10, "most_likely": 15, "pessimistic": 22},
            "risk_factors": {
                "bug_complexity": {
                    "type": "continuous",
                    "mean": 1.0,
                    "std": 0.25
                }
            },
            "dependencies": ["Desain_Arsitektur"]
        },
        "Implementasi_Backend": {
            "base_params": {"optimistic": 12, "most_likely": 18, "pessimistic": 28},
            "risk_factors": {
                "api_complexity": {
                    "type": "continuous",
                    "mean": 1.0,
                    "std": 0.3
                }
            },
            "dependencies": ["Desain_Arsitektur"]
        },
        "Pengujian_Integrasi": {
            "base_params": {"optimistic": 5, "most_likely": 8, "pessimistic": 14},
            "risk_factors": {
                "bug_discovery_rate": {
                    "type": "continuous",
                    "mean": 1.0,
                    "std": 0.3
                }
            },
            "dependencies": ["Implementasi_Frontend", "Implementasi_Backend"]
        },
        "Deployment_Produksi": {
            "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 6},
            "risk_factors": {
                "server_issues": {
                    "type": "discrete",
                    "probability": 0.1,
                    "impact": 0.5
                }
            },
            "dependencies": ["Pengujian_Integrasi"]
        }
    }
    
    # Menampilkan konfigurasi tahapan di sidebar
    for stage_name, config in default_config.items():
        with st.sidebar.expander(f"⚙️ {stage_name.replace('_', ' ')}", expanded=False):
            optimistic = st.number_input(
                    f"Optimistic",
                    min_value=1,
                    max_value=100,
                    value=config['base_params']['optimistic'],
                    key=f"opt_{stage_name}"
                )
                
            most_likely = st.number_input(
                    f"Most Likely",
                    min_value=1,
                    max_value=100,
                    value=config['base_params']['most_likely'],
                    key=f"ml_{stage_name}"
                )
                
            pessimistic = st.number_input(
                    f"Pessimistic",
                    min_value=1,
                    max_value=100,
                    value=config['base_params']['pessimistic'],
                    key=f"pes_{stage_name}"
                )
                
            
            # Update config
            default_config[stage_name]['base_params'] = {
                'optimistic': optimistic,
                'most_likely': most_likely,
                'pessimistic': pessimistic
            }
    
    # Tombol untuk menjalankan
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #666;">
    <b>Keterangan:</b><br>
    • Optimistic: Estimasi terbaik<br>
    • Most Likely: Estimasi realistis<br>
    • Pessimistic: Estimasi terburuk<br>
    • CI: Confidence Interval
    </div>
    """, unsafe_allow_html=True)
    
    # Inisialisasi session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    
    # Jalankan simulasi ketika tombol ditekan
    if run_simulation:
        with st.spinner('Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            # Inisialisasi simulator
            simulator = MonteCarloProjectSimulation(
                stages_config=default_config,
                num_simulations=num_simulations
            )
            
            # Jalankan simulasi
            results = simulator.run_simulation()
            
            # Simpan ke session state
            st.session_state.simulation_results = results
            st.session_state.simulator = simulator
            
            st.success(f'Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.')
    
    # Tampilkan hasil jika simulasi sudah dijalankan
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        simulator = st.session_state.simulator
        
        # ====================================================================
        # BAGIAN 1: STATISTIK UTAMA
        # ====================================================================
        st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)
        
        # Metrik utama
        total_duration = results['Total_Duration']
        mean_duration = total_duration.mean()
        median_duration = np.median(total_duration)
        ci_80 = np.percentile(total_duration, [10, 90])
        ci_95 = np.percentile(total_duration, [2.5, 97.5])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{mean_duration:.1f}</h3>
                <p>Rata-rata Durasi (Hari)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{median_duration:.1f}</h3>
                <p>Median Durasi (Hari)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{ci_80[0]:.1f} - {ci_80[1]:.1f}</h3>
                <p>80% Confidence Interval</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{ci_95[0]:.1f} - {ci_95[1]:.1f}</h3>
                <p>95% Confidence Interval</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # BAGIAN 2: VISUALISASI UTAMA
        # ====================================================================
        st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)
        
        # Tab untuk visualisasi
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Distribusi Durasi", 
            "🎯 Probabilitas Penyelesaian", 
            "🔍 Analisis Tahapan", 
            "📊 Analisis Risiko"
        ])
        
        with tab1:
            # Plot distribusi durasi
            fig_dist, stats = create_distribution_plot(results)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Tampilkan statistik detail
            with st.expander("📋 Detail Statistik Distribusi"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Statistik Deskriptif:**")
                    st.write(f"- Rata-rata: {stats['mean']:.1f} hari")
                    st.write(f"- Median: {stats['median']:.1f} hari")
                    st.write(f"- Standar Deviasi: {stats['std']:.1f} hari")
                    st.write(f"- Minimum: {stats['min']:.1f} hari")
                    st.write(f"- Maximum: {stats['max']:.1f} hari")
                
                with col2:
                    st.write("**Confidence Intervals:**")
                    st.write(f"- 80% CI: [{stats['ci_80'][0]:.1f}, {stats['ci_80'][1]:.1f}] hari")
                    st.write(f"- 95% CI: [{stats['ci_95'][0]:.1f}, {stats['ci_95'][1]:.1f}] hari")
        
        with tab2:
            # Plot probabilitas penyelesaian
            fig_prob = create_completion_probability_plot(results)
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Analisis probabilitas deadline
            with st.expander("📅 Analisis Probabilitas Deadline"):
                deadlines = [60, 65, 70, 75, 80]
                cols = st.columns(len(deadlines))
                
                for i, deadline in enumerate(deadlines):
                    prob_on_time = np.mean(total_duration <= deadline)
                    prob_late = 1 - prob_on_time
                    
                    with cols[i]:
                        st.metric(
                            label=f"Deadline {deadline} hari",
                            value=f"{prob_on_time:.1%}",
                            delta=f"{prob_late:.1%} terlambat" if prob_late > 0 else "Tepat waktu",
                            delta_color="inverse"
                        )
        
        with tab3:
            # Plot critical path dan boxplot
            col1, col2 = st.columns(2)
            
            with col1:
                # Critical path analysis
                critical_analysis = simulator.calculate_critical_path_probability()
                fig_critical = create_critical_path_plot(critical_analysis)
                st.plotly_chart(fig_critical, use_container_width=True)
            
            with col2:
                # Boxplot durasi per tahapan
                fig_boxplot = create_stage_boxplot(results, simulator.stages)
                st.plotly_chart(fig_boxplot, use_container_width=True)
            
            # Detail tahapan kritis
            with st.expander("🔍 Detail Analisis Critical Path"):
                critical_df = critical_analysis.sort_values('probability', ascending=False)
                st.dataframe(critical_df, use_container_width=True)
        
        with tab4:
            # Analisis risiko
            col1, col2 = st.columns(2)
            
            with col1:
                # Kontribusi risiko
                risk_contrib = simulator.analyze_risk_contribution()
                fig_risk = create_risk_contribution_plot(risk_contrib)
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                # Heatmap korelasi
                fig_corr = create_correlation_heatmap(results, simulator.stages)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Detail analisis risiko
            with st.expander("📋 Detail Analisis Kontribusi Risiko"):
                st.dataframe(risk_contrib, use_container_width=True)
        
        # ====================================================================
        # BAGIAN 3: ANALISIS STATISTIK LENGKAP
        # ====================================================================
        st.markdown('<h2 class="sub-header">📋 Analisis Statistik Lengkap</h2>', unsafe_allow_html=True)
        
        with st.expander("📊 Tabel Data Simulasi", expanded=False):
            st.dataframe(results.head(100), use_container_width=True)
        
        # Statistik per tahapan
        st.markdown("**Statistik Durasi per Tahapan:**")
        stage_stats = pd.DataFrame()
        for stage_name in simulator.stages.keys():
            stage_data = results[stage_name]
            stage_stats[stage_name] = [
                stage_data.mean(),
                stage_data.std(),
                np.percentile(stage_data, 25),
                np.percentile(stage_data, 50),
                np.percentile(stage_data, 75)
            ]
        
        stage_stats.index = ['Mean', 'Std Dev', 'Q1', 'Median', 'Q3']
        st.dataframe(stage_stats.T, use_container_width=True)
        
        # ====================================================================
        # BAGIAN 4: ANALISIS DEADLINE DAN REKOMENDASI
        # ====================================================================
        st.markdown('<h2 class="sub-header">🎯 Analisis Deadline & Rekomendasi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input deadline target
            target_deadline = st.number_input(
                "Masukkan deadline target (hari):",
                min_value=40,
                max_value=120,
                value=70,
                step=1
            )
            
            # Hitung probabilitas untuk deadline target
            prob_target = np.mean(total_duration <= target_deadline)
            days_at_risk = max(0, np.percentile(total_duration, 95) - target_deadline)
            
            st.metric(
                label=f"Probabilitas selesai dalam {target_deadline} hari",
                value=f"{prob_target:.1%}",
                delta=f"Potensi keterlambatan: {days_at_risk:.1f} hari" if days_at_risk > 0 else "Tepat waktu",
                delta_color="inverse"
            )
        
        with col2:
            # Rekomendasi buffer
            safety_buffer = np.percentile(total_duration, 80) - mean_duration
            contingency_reserve = np.percentile(total_duration, 95) - mean_duration
            
            st.markdown(f"""
            <div class="info-box">
                <h4>🏗️ Rekomendasi Manajemen Risiko:</h4>
                • <b>Safety Buffer</b> (untuk 80% confidence): <b>{safety_buffer:.1f} hari</b><br>
                • <b>Contingency Reserve</b> (untuk 95% confidence): <b>{contingency_reserve:.1f} hari</b><br><br>
                • <b>Estimasi jadwal yang direkomendasikan:</b><br>
                  {mean_duration:.1f} + {safety_buffer:.1f} = <b>{mean_duration + safety_buffer:.1f} hari</b>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # BAGIAN 5: INFORMASI TEKNIS
        # ====================================================================
        with st.expander("ℹ️ Informasi Teknis Simulasi", expanded=False):
            st.write(f"**Parameter Simulasi:**")
            st.write(f"- Jumlah iterasi: {num_simulations:,}")
            st.write(f"- Jumlah tahapan: {len(simulator.stages)}")
            st.write(f"- Seed acak: 42 (untuk hasil yang dapat direproduksi)")
            
            st.write(f"\n**Konfigurasi Tahapan:**")
            for stage_name, config in default_config.items():
                base = config['base_params']
                st.markdown(f"""
                <div class="stage-card">
                <b>{stage_name.replace('_', ' ')}</b><br>
                • Optimistic: {base['optimistic']} hari<br>
                • Most Likely: {base['most_likely']} hari<br>
                • Pessimistic: {base['pessimistic']} hari
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Tampilkan instruksi jika simulasi belum dijalankan
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background-color: #f8f9fa; border-radius: 10px;">
            <h3>🚀 Siap untuk memulai simulasi?</h3>
            <p>Atur parameter di sidebar kiri, lalu klik tombol <b>"Run Simulation"</b> untuk memulai analisis.</p>
            <p>📊 Hasil simulasi akan ditampilkan di sini setelah proses selesai.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan preview konfigurasi
        st.markdown('<h2 class="sub-header">📋 Preview Konfigurasi Tahapan</h2>', unsafe_allow_html=True)
        
        for stage_name, config in default_config.items():
            base = config['base_params']
            st.markdown(f"""
            <div class="stage-card">
            <b>{stage_name.replace('_', ' ')}</b> | 
            Optimistic: {base['optimistic']} hari | 
            Most Likely: {base['most_likely']} hari | 
            Pessimistic: {base['pessimistic']} hari
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><b>Simulasi Monte Carlo untuk Estimasi Waktu Proyek</b></p>
    <p>⚠️ Hasil simulasi ini merupakan estimasi probabilistik dan bukan prediksi pasti.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 5. JALANKAN APLIKASI
# ============================================================================
if __name__ == "__main__":
    main()