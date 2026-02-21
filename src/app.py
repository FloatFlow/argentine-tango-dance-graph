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

def load_queue_state():
    state_path = "queue_state.json"
    if not os.path.exists(state_path):
        return {"queue": [], "seen_videos": [], "seen_channels": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"queue": [], "seen_videos": [], "seen_channels": []}

# --- Expensive Computation Cache ---
@st.cache_data
def get_atlas_data(dancers_data):
    """
    Pre-computes the dataframe and clustering to avoid delays on interaction.
    Merges dancers with identical coordinates into a single 'Couple' point.
    """
    # 1. Group by Coordinates to handle overlaps
    grouped_points = {}
    
    for name, info in dancers_data.items():
        embedding = info.get("style_embedding")
        if embedding:
            # Round to 4 decimals to catch exact/near-exact overlaps
            key = (round(embedding["x"], 4), round(embedding["y"], 4))
            
            if key not in grouped_points:
                grouped_points[key] = []
            
            grouped_points[key].append({
                "name": name,
                "videos": len(info.get("videos", [])),
                "x": embedding["x"],
                "y": embedding["y"]
            })
    
    # 2. Flatten back to list, merging couples
    plot_data = []
    for key, group in grouped_points.items():
        if not group:
            continue
            
        if len(group) == 1:
            plot_data.append(group[0])
        else:
            # Sort by video count (desc) then name to be deterministic
            group.sort(key=lambda x: (-x["videos"], x["name"]))
            
            # Create composite entry
            names = [g["name"] for g in group]
            display_name = " & ".join(names[:3]) # Limit to 3 names to avoid huge strings
            if len(names) > 3:
                display_name += f" (+{len(names)-3})"
                
            # Use data from the first entity (they are overlapping, so x/y is same)
            primary = group[0]
            plot_data.append({
                "name": display_name,
                "x": primary["x"],
                "y": primary["y"],
                "videos": primary["videos"],
                # Store original names for search/filtering if needed
                "members": names 
            })
    
    if not plot_data:
        return pd.DataFrame()

    df = pd.DataFrame(plot_data)

    # 3. Clustering
    if len(df) > 10:
        # Adjusted parameters for potentially denser merged points
        hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
        df['cluster'] = hdb.fit_predict(df[['x', 'y']])
        df['cluster'] = df['cluster'].astype(str)
    else:
        df['cluster'] = "0"

    # 4. Sizing
    df['size_log'] = np.log1p(df['videos']) * 8 
    
    return df


@st.cache_data
def get_library_data(videos_data):
    """
    Flattens video dictionary into a DataFrame for the Library tab.
    """
    rows = []
    for vid_id, data in videos_data.items():
        # Extract Orchestra
        orchestra = None
        if data.get("music") and data["music"].get("orchestra"):
            orchestra = data["music"]["orchestra"]
            
        # Extract Event
        event = None
        if data.get("event") and data["event"].get("name"):
            event = data["event"]["name"]
            
        # Extract Dancers (just names for searching)
        dancers_list = []
        if data.get("performances"):
            for p in data["performances"]:
                dancers_list.extend([d.get("name") for d in p.get("dancers", [])])
        # Legacy fallback
        elif data.get("dancers"):
            dancers_list = [d.get("name") for d in data["dancers"]]
            
        rows.append({
            "id": vid_id,
            "title": data.get("title", "Untitled"),
            "orchestra": orchestra or "Unknown",
            "event": event or "Unknown",
            "year": (data.get("event") or {}).get("year"),
            "videographer": data.get("videographer", "Unknown"),
            "dancers": ", ".join(dancers_list),
            "url": data.get("url"),
            "duration": data.get("duration", 0)
        })
    
    if not rows:
        return pd.DataFrame()
        
    return pd.DataFrame(rows)


data = load_data()
videos = data.get("videos", {})
dancers = data.get("dancers", {})
queue_state = load_queue_state()

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
                        if st.button(f"{name} ({score:.2f})", key=f"modal_sim_{name}", width="stretch"):
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
tab_atlas, tab_library, tab_dashboard = st.tabs(["Style Atlas", "Video Library", "Dashboard"])

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
            if st.button("🔍 Search", width="stretch"):
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
            textfont=dict(size=10, color='rgba(200,200,200,0.9)'),
            marker=dict(opacity=0.8, line=dict(width=0)) # Default clean look
        )
        
        # Explicitly highlight the selected point using a separate trace
        # This overcomes the issue where 'update_traces' fails to map column data across multiple color traces
        if active_dancer:
            selected_row = df[df['name'] == active_dancer]
            if not selected_row.empty:
                fig.add_scatter(
                    x=selected_row['x'],
                    y=selected_row['y'],
                    mode='markers',
                    marker=dict(
                        size=30, # Match the 'final_size' logic for selected
                        color='rgba(0,0,0,0)', # Transparent fill to reveal underlying cluster color
                        line=dict(width=4, color='Red'),
                        symbol='circle'
                    ),
                    hoverinfo='skip', # Let the underlying point handle the hover text
                    showlegend=False,
                    customdata=selected_row[['name']] # Enable clicking this ring to keep selection valid
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
            width="stretch", 
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

# --- TAB 2: VIDEO LIBRARY ---
with tab_library:
    df_videos = get_library_data(videos)
    if df_videos.empty:
        st.info("No videos found yet.")
    else:
        # --- Filters ---
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        
        with c_filter1:
            all_orchestras = sorted([x for x in df_videos['orchestra'].unique() if x != "Unknown"])
            sel_orch = st.multiselect("Orchestra", options=all_orchestras)
            
        with c_filter2:
            all_events = sorted([x for x in df_videos['event'].unique() if x != "Unknown"])
            sel_event = st.multiselect("Event", options=all_events)
            
        with c_filter3:
            # Search by dancer name (string contains)
            search_dancer = st.text_input("Dancer Name", placeholder="e.g. Chicho")

        # --- Apply Filters ---
        filtered_df = df_videos.copy()
        if sel_orch:
            filtered_df = filtered_df[filtered_df['orchestra'].isin(sel_orch)]
        if sel_event:
            filtered_df = filtered_df[filtered_df['event'].isin(sel_event)]
        if search_dancer:
            filtered_df = filtered_df[filtered_df['dancers'].str.contains(search_dancer, case=False, na=False)]
            
        # Limit to 50 to prevent freezing
        display_df = filtered_df.head(50)
        
        st.markdown(f"**Showing {len(display_df)} of {len(filtered_df)} matching videos**")
        
        # --- Grid Layout ---
        # We use columns to create a grid
        cols = st.columns(3)
        for idx, row in display_df.iterrows():
            with cols[idx % 3]:
                with st.container(border=True):
                    if row['url']:
                        st.video(row['url'])
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['orchestra']} • {row['event']}")

# --- TAB 3: DASHBOARD ---
with tab_dashboard:
    st.header("Global Statistics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Videos", len(videos))
    c2.metric("Total Dancers", len(dancers))
    
    # Queue Stats
    q_len = len(queue_state.get("queue", []))
    seen_len = len(queue_state.get("seen_videos", []))
    c3.metric("Pending Queries", q_len, delta=seen_len, delta_color="off")
    
    st.divider()
    
    col_charts, col_queue = st.columns([2, 1])
    
    with col_charts:
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
            st.plotly_chart(fig, width="stretch")

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
        
    with col_queue:
        st.subheader("Discovery Queue")
        queue_list = queue_state.get("queue", [])
        if queue_list:
            st.dataframe(pd.DataFrame(queue_list, columns=["Query"]), height=400, hide_index=True)
        else:
            st.info("Queue is empty.")