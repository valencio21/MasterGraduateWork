import streamlit as st
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(layout="wide", page_title="MBA Dashboard")


SINGLE_RULES_FILES = {
    "Level 0: Filter: None": "rules_0_unfiltered.parquet",
    "Level 1: Filter: Aisle": "rules_1_aisle.parquet",
    "Level 2: Filter: Aisle + Cluster": "rules_2_aisle_cluster.parquet",
    "Level 3: Filter: Aisle + Cluster + Department": "rules_3_aisle_cluster_department.parquet"
}

MULTY_RULES_FILES = {
    "Level 0: Filter: None": "multy_rules_0_unfiltered.parquet",
    "Level 1: Filter: Aisle": "multy_rules_1_aisle.parquet",
    "Level 2: Filter: Aisle + Cluster": "multy_rules_2_aisle_cluster.parquet",
    "Level 3: Filter: Aisle + Cluster + Department": "multy_rules_3_aisle_cluster_department.parquet"
}


# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=['antecedent_names', 'consequent_names', 'support', 'confidence', 'norm_log_lift', 'norm_ipf', 'weighted_score'])

    def to_string(x):
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return ', '.join([str(i) for i in x])
        return str(x)

    df['antecedent_str'] = df['antecedent_names'].apply(to_string)
    df['consequent_str'] = df['consequent_names'].apply(to_string)
    df['rule_name'] = df['antecedent_str'] + " -> " + df['consequent_str']
    return df


# --- SIDEBAR FILTERS ---
st.sidebar.title("Rule Configuration")

# --- ДОДАНО: Головний перемикач типу правил ---
rule_type = st.sidebar.radio(
    "Select Rule Type:",
    ["Item to Item (1 -> 1)", "Itemset to Itemset (N -> M)"],
    index=0
)

# Визначаємо, який словник файлів використовувати
if rule_type == "Item to Item (1 -> 1)":
    FILTER_LEVELS = SINGLE_RULES_FILES
else:
    FILTER_LEVELS = MULTY_RULES_FILES

st.sidebar.markdown("---")
st.sidebar.title("Pipeline Stage")

selected_level = st.sidebar.radio(
    "Select Rule Filtration Level:",
    list(FILTER_LEVELS.keys()),
    index=2
)

st.sidebar.markdown("---")
st.sidebar.title("Filter Settings")

# Завантажуємо дані на основі обраного словника та рівня
current_file = FILTER_LEVELS[selected_level]
df = load_data(current_file)

if df.empty:
    st.error(f"File '{current_file}' not found. Please ensure it is in the project directory.")
    st.stop()

RULES_DFS = {}
for level_str, file_path in FILTER_LEVELS.items():
    lvl_int = int(level_str.split(":")[0].replace("Level ", ""))
    RULES_DFS[lvl_int] = load_data(file_path)

# Extract unique items
unique_items = set()
for col in ['antecedent_str', 'consequent_str']:
    if col in df.columns:
        for item_set_str in df[col].dropna().unique():
            for single_item in str(item_set_str).split(','):
                cleaned_item = single_item.strip()
                if cleaned_item:
                    unique_items.add(cleaned_item)

all_items = sorted(list(unique_items))

# Фільтри за Антецедентом та Консеквентом
selected_antecedents = st.sidebar.multiselect(
    "Select Target Item(s) (Antecedent):",
    options=all_items,
    help="You can select multiple items. The rule must contain ALL selected items."
)

selected_consequents = st.sidebar.multiselect(
    "Select Target Item(s) (Consequent):",
    options=all_items,
    help="You can select multiple items. The rule must contain ALL selected items."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Metric Filters")

# 1. Знаходимо динамічні максимуми (з предохранителями від пустих датафреймів)
max_conf = float(df['confidence'].max()) if ('confidence' in df.columns and not df.empty) else 1.0
max_conf = max(0.01, max_conf)  # max_value має быть строго більше min_value (0.0)

max_score = float(df['weighted_score'].max()) if ('weighted_score' in df.columns and not df.empty) else 1.0
max_score = max(0.01, max_score)

max_sup = float(df['support'].max()) if ('support' in df.columns and not df.empty) else 0.1
max_sup = max(0.0002, max_sup)  # min_value у нас 0.0001, тому max_value має бути хоча б 0.0002

# 2. Метричні фільтри з динамічними max_value та безпечними дефолтними значеннями
min_conf = st.sidebar.slider(
    "Minimum Confidence:", 
    min_value=0.0, 
    max_value=max_conf, 
    value=min(0.1, max_conf),  # Дефолт 0.1, але не більше максимума
    step=0.01
)

min_score = st.sidebar.slider(
    "Minimum Weighted Score:", 
    min_value=0.0, 
    max_value=max_score, 
    value=0.0, 
    step=0.001,
    format="%.4f"
)

min_support = st.sidebar.slider(
    "Minimum Support:", 
    min_value=0.0001, 
    max_value=max_sup, 
    value=0.0001, 
    step=0.0001, 
    format="%.4f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sorting & Diversity Limits")

# 1. Динамічне сортування
sort_metric = st.sidebar.selectbox(
    "Sort Rules By (determines the 'Best' rules):",
    ["weighted_score", "confidence", "norm_log_lift", "norm_ipf"]
)

# 2. Фільтри різноманітності (Diversity limits)
max_rules_in_df = max(1, len(df))
default_div_limit = min(10, max_rules_in_df)

max_per_antecedent = st.sidebar.number_input(
    "Max Rules per Antecedent:", 
    min_value=1, 
    max_value=max_rules_in_df, 
    value=default_div_limit, 
    step=1
)

max_per_consequent = st.sidebar.number_input(
    "Max Rules per Consequent:", 
    min_value=1, 
    max_value=max_rules_in_df, 
    value=default_div_limit, 
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Limits")

default_table_limit = min(50, max_rules_in_df)
table_limit = st.sidebar.number_input(
    "Number of rules in Table:", 
    min_value=1, 
    max_value=max_rules_in_df, 
    value=default_table_limit, 
    step=10
)

# --- FILTERING DATA ---
filtered_df = df.copy()

# 1. Фільтрація за назвами
if selected_antecedents:  # Якщо користувач обрав хоча б один товар
    for item in selected_antecedents:
        filtered_df = filtered_df[filtered_df['antecedent_str'].str.contains(item, regex=False, case=False)]

if selected_consequents:
    for item in selected_consequents:
        filtered_df = filtered_df[filtered_df['consequent_str'].str.contains(item, regex=False, case=False)]

# 2. Фільтрація за метриками (мінімуми)
mask = (filtered_df['confidence'] >= min_conf) & (filtered_df['weighted_score'] >= min_score)
if 'support' in filtered_df.columns:
    mask = mask & (filtered_df['support'] >= min_support)
filtered_df = filtered_df[mask]

# 3. ДИНАМІЧНЕ СОРТУВАННЯ (за обраною метрикою)
filtered_df = filtered_df.sort_values(by=sort_metric, ascending=False)

# 4. ЗАСТОСУВАННЯ ЛІМІТІВ (Diversity)
# Беремо Топ-N найкращих правил для кожного антецедента
filtered_df = filtered_df.groupby('antecedent_str').head(max_per_antecedent)

# З того, що залишилось, беремо Топ-N найкращих для кожного консеквента
filtered_df = filtered_df.groupby('consequent_str').head(max_per_consequent)


# --- MAIN DASHBOARD ---
st.title("Association Rules Analysis")
st.write(f"Total rules found matching filters: **{len(filtered_df)}**")


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Table & Overview",
    "🕸️ Network Topology",
    "🎯 Bar Charts & Heatmap",
    "🎛️ Advanced Metrics"
])


with tab1:
    # 1. TABLE
    st.subheader("Top Rules Table")
    st.dataframe(
        filtered_df[['antecedent_str', 'consequent_str', 'confidence', 'norm_log_lift', 'norm_ipf', 'weighted_score']].head(
            table_limit))


with tab2:
    # 3. NETWORK GRAPH (Interactive PyVis)
    st.subheader("Items Association Network (Interactive)")

    # Вибір режиму графа
    graph_mode = st.radio(
        "Select Graph Mode:",
        ["Default (Global Network)", "Ego-centric (Targeted Network)"],
        horizontal=True
    )

    # --- ДОДАНО: Динамічний вибір метрики для побудови та товщини графа ---
    metric_options = {
        "Weighted Score": "weighted_score",
        "Confidence": "confidence",
        "Lift": "norm_log_lift",
        "Support": "support"
    }

    # Перевірка на наявність колонки support
    if 'support' not in df.columns:
        metric_options.pop("Support", None)

    selected_edge_metric = st.selectbox(
        "Select Metric for Edge Weight & Branching Priority:",
        list(metric_options.keys()),
        index=0
    )
    edge_col = metric_options[selected_edge_metric]


    # Допоміжна функція для динамічного масштабування товщини стрілок (від 1 до 10 пікселів)
    def get_scaled_width(val, min_val, max_val):
        if max_val == min_val:
            return 5
        norm = (val - min_val) / (max_val - min_val)
        return int(1 + norm * 9)


    if graph_mode == "Default (Global Network)":
        if not filtered_df.empty:
            graph_limit = st.slider(
                "Number of rules to show in Network Graph:",
                min_value=10, max_value=200, value=30, step=10,
                key="network_graph_slider"
            )

            # ЗМІНЕНО: Сортуємо датафрейм спеціально для графа за обраною метрикою
            net_df = filtered_df.sort_values(by=edge_col, ascending=False).head(graph_limit)

            min_metric = net_df[edge_col].min()
            max_metric = net_df[edge_col].max()

            net = Network(height='600px', width='100%', bgcolor='#0E1117', font_color='white', directed=True)
            net.repulsion(node_distance=150, spring_length=150)

            for _, row in net_df.iterrows():
                src = row['antecedent_str']
                dst = row['consequent_str']
                metric_val = row[edge_col]

                net.add_node(src, label=src, title=src, size=15, color='#4CAF50')
                net.add_node(dst, label=dst, title=dst, size=15, color='#2196F3')

                # Масштабована товщина
                edge_width = get_scaled_width(metric_val, min_metric, max_metric)
                tooltip = f"{selected_edge_metric}: {metric_val:.4f}"

                net.add_edge(src, dst, value=metric_val, width=edge_width, title=tooltip, color='#888888')

            try:
                net.save_graph('network_graph.html')
                with open('network_graph.html', 'r', encoding='utf-8') as HtmlFile:
                    components.html(HtmlFile.read(), height=620)
            except Exception as e:
                st.error(f"Error rendering network graph: {e}")
        else:
            st.info("No rules found for the Default Graph with current filters.")

    else:
        # --- ЛОГІКА ЕГО-ЦЕНТРИЧНОЇ МЕРЕЖІ ---
        if not selected_antecedents:
            st.warning("Please select at least one item in 'Target Item(s) (Antecedent)' to build an Ego-centric network.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ego_depth = st.slider("Depth (Levels of connection):", min_value=1, max_value=3, value=2, step=1)
            with col2:
                ego_branch = st.slider("Branching Factor (Max connections per item):", min_value=1, max_value=10, value=3,
                                       step=1)

            ego_edges = []
            current_sources = set(selected_antecedents)
            seen_edges = set()

            for d in range(ego_depth):
                next_sources = set()

                for src in current_sources:
                    found_for_src = 0

                    for lvl in [3, 2, 1, 0]:
                        level_df = RULES_DFS.get(lvl)
                        if level_df is None or level_df.empty:
                            continue

                        mask = (level_df['antecedent_str'] == src) & \
                               (level_df['confidence'] >= min_conf) & \
                               (level_df['weighted_score'] >= min_score)

                        if 'support' in level_df.columns:
                            mask &= (level_df['support'] >= min_support)

                        # ЗМІНЕНО: Відбираємо найкращі правила (Branching) за обраною метрикою
                        src_rules = level_df[mask].sort_values(by=edge_col, ascending=False)

                        for _, row in src_rules.iterrows():
                            con = row['consequent_str']
                            metric_val = row[edge_col]
                            edge_key = (src, con)

                            if edge_key not in seen_edges:
                                ego_edges.append((src, con, metric_val, d, lvl))
                                seen_edges.add(edge_key)
                                next_sources.add(con)
                                found_for_src += 1

                            if found_for_src >= ego_branch:
                                break

                        if found_for_src >= ego_branch:
                            break

                current_sources = next_sources
                if not current_sources:
                    break

            if ego_edges:
                all_metric_vals = [e[2] for e in ego_edges]
                min_metric = min(all_metric_vals)
                max_metric = max(all_metric_vals)

                net = Network(height='600px', width='100%', bgcolor='#0E1117', font_color='white', directed=True)
                net.repulsion(node_distance=200, spring_length=200)

                colors = ['#E91E63', '#FF9800', '#4CAF50', '#00BCD4']

                for ant, con, metric_val, depth_lvl, rule_lvl in ego_edges:
                    is_root = ant in selected_antecedents

                    ant_color = colors[0] if is_root else colors[min(depth_lvl, len(colors) - 1)]
                    con_color = colors[min(depth_lvl + 1, len(colors) - 1)]
                    ant_size = 25 if is_root else 15

                    net.add_node(ant, label=ant, title=ant, size=ant_size, color=ant_color)
                    net.add_node(con, label=con, title=con, size=15, color=con_color)

                    # ЗМІНЕНО: Масштабована товщина
                    edge_width = get_scaled_width(metric_val, min_metric, max_metric)
                    tooltip_text = f"{selected_edge_metric}: {metric_val:.4f} (Level {rule_lvl})"

                    net.add_edge(ant, con, value=metric_val, width=edge_width, title=tooltip_text, color='#888888')

                try:
                    net.save_graph('network_ego.html')
                    with open('network_ego.html', 'r', encoding='utf-8') as HtmlFile:
                        components.html(HtmlFile.read(), height=620)
                except Exception as e:
                    st.error(f"Error rendering ego network graph: {e}")
            else:
                st.info("No deeper connections found. Try lowering Confidence or Score thresholds.")


with tab3:
    # 2. BAR CHART
    st.subheader("Top Recommended Items (Bar Chart)")

    if selected_antecedents and not filtered_df.empty:

        bar_chart_limit = st.slider(
            "Number of recommendations to show in chart:",
            min_value=5, max_value=50, value=10, step=5,
            key="bar_chart_slider"
        )

        top_bar_rules = filtered_df.head(bar_chart_limit)

        fig_bar = px.bar(
            top_bar_rules,
            x="weighted_score",
            y="consequent_str",
            orientation='h',
            color="confidence",
            title=f"Top {bar_chart_limit} Recommendations for: {', '.join(selected_antecedents)}",
            labels={"consequent_str": "Recommended Item", "weighted_score": "Weighted Score"}
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("Select a specific item in the sidebar to see top recommendations for it.")


    # 2.1. HEATMAP (Матриця крос-селу)
    st.subheader("Cross-Sell Matrix (Heatmap)")
    st.markdown("*Shows the Weighted Score intensity between top Antecedents and Consequents.*")
    if not filtered_df.empty:
        heatmap_df = filtered_df.head(50).copy()

        max_chars = 35
        heatmap_df['ant_short'] = heatmap_df['antecedent_str'].apply(
            lambda x: x[:max_chars] + '...' if len(x) > max_chars else x
        )
        heatmap_df['con_short'] = heatmap_df['consequent_str'].apply(
            lambda x: x[:max_chars] + '...' if len(x) > max_chars else x
        )

        pivot_df = heatmap_df.pivot_table(index='ant_short', columns='con_short', values='weighted_score', aggfunc='max')

        fig_heat = px.imshow(
            pivot_df,
            text_auto=".4f",
            color_continuous_scale='Plasma',
            title="Weighted Score Heatmap (Top 50 Rules)",
            labels=dict(x="Consequent (Recommended)", y="Antecedent (In Cart)", color="Score")
        )
        fig_heat.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heat, use_container_width=True)


with tab4:
    # --- ДОДАНО: QUADRANT ANALYSIS MATRIX (Scatter Plot) ---
    st.subheader("Матриця квадрантного аналізу: Confidence vs. Lift")
    st.markdown("""
        *Матриця квадрантного аналізу розподіляє правила асоціації за чотирма категоріями (перехрестя медіан):*
        * 📈 **Надійні та сильні асоціації (Top-Right):** Висока абсолютна ймовірність (Confidence) та сильний відносний взаємозв'язок (Lift). Правила для формування стабільних пакетних пропозицій.
        * ✨ **Нішеві пари з високою синергією (Top-Left):** Низька абсолютна ймовірність, але сильний відносний взаємозв'язок. Потенційні драйвери крос-продажів у специфічних сегментах, які інакше були б непомітними.
        * 🔄 **Тривіальні або глобально популярні асоціації (Bottom-Right):** Висока абсолютна ймовірність, але слабкий відносний взаємозв'язок. Товари часто купують разом лише тому, що Товар B є глобально популярним сам по собі.
        * ⚪ **Слабкі асоціації (Bottom-Left):** Низька ймовірність та слабкий відносний взаємозв'язок. Правила з мінімальною цінністю.
        """)

    if not filtered_df.empty:
        has_support = 'support' in filtered_df.columns
        size_col = 'support' if has_support else None

        # Динамічні медіани по ПОТОЧНІЙ вибірці
        conf_mid = filtered_df['confidence'].median()
        lift_mid = filtered_df['norm_log_lift'].median()

        fig_scatter = px.scatter(
            filtered_df,
            x="confidence",
            y="norm_log_lift",
            size=size_col,
            color="weighted_score",
            hover_name="rule_name",
            hover_data={
                "confidence": ":.4f",
                "norm_log_lift": ":.4f",
                "norm_ipf": ":.4f",
                "weighted_score": ":.4f",
                "support": ":.6f" if has_support else False
            },
            # --- ЗМІНЕНО: Висококонтрастна палітра для темного фону, що підкреслює малі зміни ---
            color_continuous_scale=px.colors.sequential.Inferno,
            size_max=25,
            title="Performance Distribution by Metric Quadrants",
            labels={
                "confidence": "Confidence (Probability)",
                "norm_log_lift": "Normalized Lift (Synergy)",
                "weighted_score": "Weighted Score",
                "support": "Support (Volume)"
            }
        )

        fig_scatter.add_hline(
            y=lift_mid,
            line_dash="dot",
            line_color="gray",
            annotation_text="Median Lift",
            annotation_position="bottom right"
        )
        fig_scatter.add_vline(
            x=conf_mid,
            line_dash="dot",
            line_color="gray",
            annotation_text="Median Conf.",
            annotation_position="top left"
        )

        fig_scatter.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No rules found to display the Quadrant Analysis.")

    # 2.2. PARALLEL COORDINATES (Багатовимірний аналіз)
    st.subheader("Metrics Parallel Coordinates Plot")
    st.markdown("*Visualizes how Support, Confidence, Lift, and IPF combine to form the final Weighted Score.*")
    if not filtered_df.empty and len(filtered_df) > 1:
        cols_to_plot = ['support', 'confidence', 'norm_log_lift', 'norm_ipf', 'weighted_score']
        cols_to_plot = [c for c in cols_to_plot if c in filtered_df.columns]

        fig_parc = px.parallel_coordinates(
            filtered_df.head(200),
            dimensions=cols_to_plot,
            color="weighted_score",
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Parallel Coordinates (Top 200 Rules)",
            labels={
                "support": "Support",
                "confidence": "Confidence",
                "norm_log_lift": "Norm Lift",
                "norm_ipf": "Norm IPF",
                "weighted_score": "Score"
            }
        )

        fig_parc.update_layout(
            margin=dict(l=80, r=60, t=100, b=50),
            title_y=0.98
        )

        for dim in fig_parc.data[0].dimensions:
            dim.tickformat = ".4f"

        st.plotly_chart(fig_parc, use_container_width=True)


    # 2.3. ВІДФІЛЬТРОВАНИЙ РОЗПОДІЛ (Histogram)
    st.subheader("Distribution of Weighted Score (Current Filter)")
    st.markdown("*Shows the density of the final metric for the currently filtered rules.*")
    if not filtered_df.empty:
        fig_hist = px.histogram(
            filtered_df,
            x="weighted_score",
            nbins=50,
            marginal="box",  # Додає boxplot зверху
            color_discrete_sequence=['#8A2BE2'],
            title="Histogram of Weighted Score",
            labels={"weighted_score": "Weighted Score [0, 1]"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)