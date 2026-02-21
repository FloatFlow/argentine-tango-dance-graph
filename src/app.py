import streamlit as st
import json
import os
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import HDBSCAN

# Page Config
st.set_page_config(
    page_title="Tango Graph Explorer",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Tweaks for Fullscreen Feel ---
st.markdown("""
    <style>
        /* Aggressively remove padding to maximize chart space */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        /* Hide the default Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Tabs adjustment */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            padding-top: 0px;
            padding-bottom: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    db_path = "graph_db.json"
    if not os.path.exists(db_path):
        return {"videos": {}, "dancers": {}}
    
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        return {"videos": {}, "dancers": {}}

# --- Expensive Computation Cache ---
@st.cache_data
def get_atlas_data(dancers_data):
    """
    Pre-computes the dataframe and clustering to avoid delays on interaction.
    """
    plot_data = []
    for name, info in dancers_data.items():
        embedding = info.get("style_embedding")
        if embedding:
            plot_data.append({
                "name": name,
                "x": embedding["x"],
                "y": embedding["y"],
                "videos": len(info.get("videos", [])),
            })
    
    if not plot_data:
        return pd.DataFrame()

    df = pd.DataFrame(plot_data)

    # 1. Clustering
    if len(df) > 10:
        hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
        df['cluster'] = hdb.fit_predict(df[['x', 'y']])
        df['cluster'] = df['cluster'].astype(str)
    else:
        df['cluster'] = "0"

    # 2. Sizing
    df['size_log'] = np.log1p(df['videos']) * 8 
    
    return df


data = load_data()
videos = data.get("videos", {})
dancers = data.get("dancers", {})

# --- Helper Functions ---
def format_duration(seconds):
    if not seconds:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# --- Modal: Search ---
@st.dialog("Search Dancer")
def show_search_modal(all_names):
    st.caption("Selecting a dancer will highlight them on the map.")
    
    # No callback needed. We just check the return value.
    selected = st.selectbox(
        "Type to search...", 
        options=all_names, 
        index=None,
        key="search_modal_box",
        placeholder="Name..."
    )

    if selected:
        st.session_state["selected_dancer"] = selected
        st.rerun()

# --- Modal: Dancer Profile ---
@st.dialog("Dancer Profile", width="large")
def show_dancer_details(dancer_name):
    if dancer_name and dancer_name in dancers:
        info = dancers[dancer_name]
        
        # --- Top Section: Stats & Related ---
        c_stats, c_related = st.columns([1, 1], gap="large")
        
        with c_stats:
            st.header(dancer_name)
            # Metrics in a single row
            m1, m2, m3 = st.columns(3)
            m1.metric("Videos", len(info.get("videos", [])))
            m2.metric("Events", len(info.get("events", {})))
            m3.metric("Partners", len(info.get("partners", {})))

        with c_related:
            st.subheader("Related Dancers")
            similar = info.get("similar_dancers", [])
            if similar:
                # Display as a dense grid of buttons
                sim_cols = st.columns(2)
                for idx, item in enumerate(similar[:6]): # Show top 6
                    name = item.get("name")
                    score = item.get("score", 0)
                    with sim_cols[idx % 2]:
                        if st.button(f"{name} ({score:.2f})", key=f"modal_sim_{name}", use_container_width=True):
                            st.session_state["selected_dancer"] = name
                            st.rerun()
            else:
                st.caption("No similar dancers found.")

        st.divider()
        
        # --- Bottom Section: Videos ---
        st.subheader("Performance History")
        video_ids = info.get("videos", [])
        
        # 2-Column Grid for Videos
        v_cols = st.columns(2)
        
        for i, vid_id in enumerate(video_ids):
            v = videos.get(vid_id)
            if not v: 
                continue
                
            with v_cols[i % 2]:
                with st.container(border=True):
                    # Native Streamlit Video Component
                    if v.get("url"):
                        st.video(v["url"])
                    
                    # Metadata
                    event_data = v.get('event') or {}
                    event_name = event_data.get('name', 'Unknown Event')
                    title = v.get('title', 'Untitled')
                    
                    st.markdown(f"**{title}**")
                    st.caption(f"📍 {event_name} • ⏱️ {format_duration(v.get('duration', 0))}")

# --- Main Layout ---
st.markdown("### 🇦🇷 TangoGraph")

# Top-level Tabs
tab_atlas, tab_dashboard = st.tabs(["Style Atlas", "Dashboard"])

# --- TAB 1: STYLE ATLAS ---
with tab_atlas:
    # Optimized Data Loading
    df = get_atlas_data(dancers)
    
    if df.empty:
        st.warning("No style embeddings found. Wait for the maintenance cycle to run.")
    else:
        # --- Controls Row ---
        c_search, c_info = st.columns([1, 5])
        with c_search:
            if st.button("🔍 Search", use_container_width=True):
                all_names = sorted(df['name'].tolist())
                show_search_modal(all_names)
        
        with c_info:
            # Display currently selected dancer info inline
            active_dancer = st.session_state.get("selected_dancer", None)
            if active_dancer:
                st.info(f"Selected: **{active_dancer}** (Click dot for details)", icon="📍")

        # --- Selection Logic ---
        active_dancer = st.session_state.get("selected_dancer", None)
        
        # Highlighting logic for the chart
        df['status'] = df['name'].apply(lambda x: 'Selected' if x == active_dancer else 'Normal')
        df['final_size'] = df.apply(lambda row: 30 if row['name'] == active_dancer else row['size_log'], axis=1)
        df['border_width'] = df['name'].apply(lambda x: 3 if x == active_dancer else 0)
        df['border_color'] = df['name'].apply(lambda x: 'Red' if x == active_dancer else 'DarkSlateGrey')
        
        # Sort so selected is on top
        df = df.sort_values(by='status', ascending=True)

        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            text='name', 
            size='final_size', 
            color='cluster', 
            hover_data=['name', 'videos'],
            # Fixed height to fit standard laptop screens without scrolling
            height=650,
            # Use custom_data to pass the name safely for click events
            custom_data=['name']
        )
        
        # Text Styling: Small font, only visible if space allows (Plotly default behavior)
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=10, color='rgba(200,200,200,0.9)')
        )
        
        # Custom Marker borders for selection highlight
        fig.update_traces(
            marker=dict(
                line=dict(
                    width=df['border_width'],
                    color=df['border_color']
                ),
                opacity=0.8
            )
        )

        # --- Auto-Centering Logic ---
        # If a dancer is selected, zoom/pan to their location
        x_range = None
        y_range = None
        
        if active_dancer:
            target_row = df[df['name'] == active_dancer]
            if not target_row.empty:
                tx = target_row.iloc[0]['x']
                ty = target_row.iloc[0]['y']
                # Create a small window around the point
                zoom_span = 2.0  
                x_range = [tx - zoom_span, tx + zoom_span]
                y_range = [ty - zoom_span, ty + zoom_span]

        # Clean Layout
        fig.update_layout(
            clickmode='event+select',
            dragmode='pan', 
            hovermode='closest',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Render Chart
        selection = st.plotly_chart(
            fig, 
            use_container_width=True, 
            on_select="rerun",
            selection_mode="points",
            config={'scrollZoom': True, 'displayModeBar': True}
        )

        # Interaction Handler
        # 1. Check for Map Click
        if selection and selection.get("selection") and len(selection["selection"]["points"]) > 0:
            clicked_name = selection["selection"]["points"][0]["customdata"][0]
            
            # Update state if changed OR if user just clicked same point (to open modal)
            st.session_state["selected_dancer"] = clicked_name
            st.session_state["show_modal"] = True
            st.rerun()
        
        # 2. Open Modal if Flag is Set
        if st.session_state.get("show_modal", False) and active_dancer:
            show_dancer_details(active_dancer)

# --- TAB 2: DASHBOARD ---
with tab_dashboard:
    st.header("Global Statistics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Videos", len(videos))
    c2.metric("Total Dancers", len(dancers))
    
    def plot_sorted_bar(data_list, title, color):
        if not data_list:
            return
        counts = Counter(data_list)
        top_items = counts.most_common(40)
        if not top_items: return
            
        df_counts = pd.DataFrame(top_items, columns=['Name', 'Count'])
        
        fig = px.bar(
            df_counts, 
            x='Name', 
            y='Count', 
            title=title,
            color_discrete_sequence=[color]
        )
        fig.update_xaxes(categoryorder='total descending')
        fig.update_layout(
            yaxis=dict(fixedrange=True), 
            xaxis=dict(fixedrange=False),
            dragmode='pan'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 1. Top Festivals
    all_events = []
    for v in videos.values():
        evt = v.get("event")
        if evt and evt.get("name"):
            all_events.append(evt["name"])
    
    plot_sorted_bar(all_events, "Top Festivals (Pareto)", "#EF553B")

    # 2. Top Videographers
    all_videographers = []
    for v in videos.values():
        if v.get("videographer"):
            all_videographers.append(v["videographer"])
            
    plot_sorted_bar(all_videographers, "Top Videographers", "#00CC96")