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
        /* Remove top padding to maximize chart space */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Hide the default Streamlit header */
        header {visibility: hidden;}
        /* Fix the main container height */
        .main .block-container {
            max-width: 100%;
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

data = load_data()
videos = data.get("videos", {})
dancers = data.get("dancers", {})

# --- Helper Functions ---
def format_duration(seconds):
    if not seconds:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# --- Modal Component ---
@st.dialog("Dancer Profile", width="large")
def show_dancer_details(dancer_name):
    if dancer_name and dancer_name in dancers:
        info = dancers[dancer_name]
        
        # Header Stats
        c1, c2 = st.columns(2)
        c1.metric("Videos", len(info.get("videos", [])))
        c2.metric("Events", len(info.get("events", {})))
        
        st.caption(f"Partners: {len(info.get('partners', {}))}")
        
        # Tabs for details inside the modal
        tab_sim, tab_vid = st.tabs(["Similar Dancers", "Videos"])
        
        with tab_sim:
            similar = info.get("similar_dancers", [])
            if similar:
                for item in similar[:8]:
                    name = item.get("name")
                    score = item.get("score", 0)
                    # Navigation inside modal
                    if st.button(f"{name} ({score:.2f})", key=f"modal_btn_{name}"):
                        st.session_state["selected_dancer"] = name
                        st.rerun()
            else:
                st.info("No similarity data yet.")
                
        with tab_vid:
            video_ids = info.get("videos", [])
            # Show recent videos
            for vid_id in video_ids[:15]:
                v = videos.get(vid_id)
                if v:
                    with st.container(border=True):
                        c_thumb, c_info = st.columns([1, 2])
                        with c_thumb:
                            if v.get("thumbnail"):
                                st.image(v["thumbnail"], use_container_width=True)
                        with c_info:
                            st.markdown(f"**[{v.get('title')}]({v.get('url')})**")
                            event_data = v.get('event') or {}
                            event_name = event_data.get('name', 'Unknown Event')
                            st.caption(f"{event_name}")
                            st.caption(f"⏱️ {format_duration(v.get('duration', 0))}")

# --- Main Layout ---
# Using tabs for top-level navigation instead of sidebar
tab_atlas, tab_dashboard = st.tabs(["Style Atlas", "Dashboard"])

# --- TAB 1: STYLE ATLAS ---
with tab_atlas:
    # Prepare Data
    plot_data = []
    for name, info in dancers.items():
        embedding = info.get("style_embedding")
        if embedding:
            plot_data.append({
                "name": name,
                "x": embedding["x"],
                "y": embedding["y"],
                "videos": len(info.get("videos", [])),
            })
    
    if not plot_data:
        st.warning("No style embeddings found. Wait for the maintenance cycle to run.")
    else:
        df = pd.DataFrame(plot_data)

        # Clustering for Color
        if len(df) > 10:
            hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
            df['cluster'] = hdb.fit_predict(df[['x', 'y']])
            df['cluster'] = df['cluster'].astype(str)
        else:
            df['cluster'] = "0"

        # Sizing
        df['size_log'] = np.log1p(df['videos']) * 8 

        # Handle Selection from Session State
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
            text='name',  # Always show names
            size='final_size', 
            color='cluster', 
            hover_data=['name', 'videos'],
            height=750 # Taller for fullscreen feel
        )
        
        # Text Styling: Small font to reduce clutter, but visible
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=9, color='rgba(200,200,200,0.8)')
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

        # Clean Layout
        fig.update_layout(
            clickmode='event+select',
            dragmode='pan', 
            hovermode='closest',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Render Chart
        # on_select="rerun" is crucial for the interaction
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
            point_index = selection["selection"]["points"][0]["point_index"]
            clicked_name = df.iloc[point_index]["name"]
            
            if clicked_name != active_dancer:
                st.session_state["selected_dancer"] = clicked_name
                st.rerun()
        
        # 2. Open Modal if Dancer Selected
        if active_dancer and active_dancer in dancers:
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