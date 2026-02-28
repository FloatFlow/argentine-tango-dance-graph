import streamlit as st
import json
import os
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import HDBSCAN
import seaborn as sns
import networkx as nx
from scipy.spatial import Delaunay
import umap
from sklearn.feature_extraction.text import TfidfTransformer

# Page Config
st.set_page_config(
    page_title="Tango Graph Explorer",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Tweaks for Fullscreen Feel & Mobile ---
st.markdown("""
    <style>
        /* Aggressively remove padding to maximize chart space, but keep enough for touch handling */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100% !important;
        }
        /* Hide the default Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom Tab Styling - Mobile Friendly */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid #444;
            flex-wrap: wrap; /* Allow wrapping on very small screens */
        }
        .stTabs [data-baseweb="tab"] {
            min-height: 45px;
            height: auto; /* Allow growth */
            white-space: pre-wrap;
            background-color: #262730;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding: 8px 16px;
            border: 1px solid #444;
            border-bottom: none;
            color: #fafafa;
            font-size: 0.9rem;
            flex-grow: 1; /* Stretch tabs to fill width on mobile */
            text-align: center;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e1117 !important;
            border-top: 3px solid #ff4b4b;
            border-bottom: 1px solid #0e1117;
            font-weight: bold;
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
def get_atlas_data(dancers_data, layout_neighbors=15, cluster_neighbors=15, min_dist=1.0, spread=5.0, init_mode='random', decoupling=False):
    """
    Computes UMAP embeddings on the fly.
    Args:
        decoupling (bool): If True, runs UMAP twice. Once for Layout (Visual) and once for Clustering (Color).
                            This allows for 'Low Neighbors' layout (tight, clean) with 'High Neighbors' clustering (global structure).
    """
    # 1. Feature Extraction
    active_dancers = [d for d, info in dancers_data.items() if len(info.get('videos', [])) > 1]
    
    if len(active_dancers) < 10:
        return pd.DataFrame(), {}

    # Build Feature Space
    all_features = set()
    for d in active_dancers:
        all_features.update(dancers_data[d].get("partners", {}).keys())
        all_features.update(dancers_data[d].get("events", {}).keys())
        all_features.update(dancers_data[d].get("orchestras", {}).keys())
    
    feature_list = sorted(list(all_features))
    feature_map = {name: i for i, name in enumerate(feature_list)}
    
    # Build Matrix
    data_matrix = []
    for dancer in active_dancers:
        row = [0] * len(feature_list)
        d_data = dancers_data[dancer]
        for p, count in d_data.get("partners", {}).items():
            if p in feature_map: row[feature_map[p]] = count
        for e, count in d_data.get("events", {}).items():
            if e in feature_map: row[feature_map[e]] = count * 2
        for o, count in d_data.get("orchestras", {}).items():
            if o in feature_map: row[feature_map[o]] = count * 1.5
        data_matrix.append(row)
        
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(data_matrix)
    
    # --- 2. UMAP Calculation (Layout) ---
    reducer_layout = umap.UMAP(
        n_neighbors=layout_neighbors, 
        n_components=2, 
        min_dist=min_dist, 
        spread=spread,
        metric='cosine',
        init=init_mode, 
        random_state=42
    )
    layout_coords = reducer_layout.fit_transform(tfidf_matrix)
    
    # --- 3. Clustering Logic ---
    if decoupling and cluster_neighbors != layout_neighbors:
        # Dual UMAP: Run a second pass purely for finding global structure clusters
        reducer_cluster = umap.UMAP(
            n_neighbors=cluster_neighbors,
            n_components=2,
            min_dist=0.0, # Tight packing for clustering
            metric='cosine',
            random_state=42
        )
        cluster_coords = reducer_cluster.fit_transform(tfidf_matrix)
    else:
        cluster_coords = layout_coords

    # Map back to dancer names
    temp_embeddings = {}
    for idx, dancer in enumerate(active_dancers):
        temp_embeddings[dancer] = {
            "x": float(layout_coords[idx][0]),
            "y": float(layout_coords[idx][1]),
            "cx": float(cluster_coords[idx][0]), # Clustering coordinates
            "cy": float(cluster_coords[idx][1])
        }

    # 4. Group by Layout Coordinates (Couples)
    grouped_points = {}
    for name in active_dancers:
        embedding = temp_embeddings.get(name)
        if embedding:
            # Group by Visual Location (x, y)
            key = (round(embedding["x"], 4), round(embedding["y"], 4))
            if key not in grouped_points:
                grouped_points[key] = []
            grouped_points[key].append({
                "name": name,
                "videos": len(dancers_data[name].get("videos", [])),
                "x": embedding["x"],
                "y": embedding["y"],
                "cx": embedding["cx"],
                "cy": embedding["cy"]
            })
    
    # 5. Flatten back to list
    plot_data = []
    for key, group in grouped_points.items():
        if not group: continue
        if len(group) == 1:
            plot_data.append(group[0])
        else:
            group.sort(key=lambda x: (-x["videos"], x["name"]))
            names = [g["name"] for g in group]
            display_name = " & ".join(names[:3])
            if len(names) > 3: display_name += f" (+{len(names)-3})"
            primary = group[0]
            plot_data.append({
                "name": display_name,
                "x": primary["x"],
                "y": primary["y"],
                "cx": primary["cx"],
                "cy": primary["cy"],
                "videos": primary["videos"],
                "members": names 
            })
    
    if not plot_data:
        return pd.DataFrame(), {}

    df = pd.DataFrame(plot_data)

    # 6. HDBSCAN Clustering
    default_palette = sns.color_palette().as_hex()
    color_map = {'-1': default_palette[7]}
    
    if len(df) > 10:
        # Cluster on the 'c' coordinates (Global Structure or Local, depending on decoupling)
        hdb = HDBSCAN(min_cluster_size=6, min_samples=2)
        try:
            df['cluster'] = hdb.fit_predict(df[['cx', 'cy']].to_numpy())
            df['cluster'] = df['cluster'].astype(str)
        except Exception as e:
            st.warning(f"Clustering failed: {e}")
            df['cluster'] = "0"
        
        # Golden Angle Coloring (Sorted by Layout X to keep rainbow coherent visually)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != '-1'])
        n_clusters = len(unique_clusters)
        if n_clusters > 0:
            cluster_centroids = []
            for c in unique_clusters:
                # Sort colors based on where they appear on the visual map (x), not the cluster map
                centroid_x = df.loc[df['cluster'] == c, 'x'].mean()
                cluster_centroids.append((c, centroid_x))
            cluster_centroids.sort(key=lambda x: x[1])
            sorted_labels = [x[0] for x in cluster_centroids]
            
            import colorsys
            import matplotlib.colors
            golden_ratio_conjugate = 0.618033988749895
            palette = []
            for i in range(n_clusters):
                h = (0.0 + i * golden_ratio_conjugate) % 1.0
                if i % 2 == 0:
                    l, s = 0.45, 0.85
                else: 
                    l, s = 0.65, 0.75
                rgb = colorsys.hls_to_rgb(h, l, s)
                palette.append(matplotlib.colors.to_hex(rgb))
            
            for i, label in enumerate(sorted_labels):
                color_map[label] = palette[i]
    else:
        df['cluster'] = "0"
        color_map['0'] = default_palette[0]

    df['size_log'] = np.log1p(df['videos']) * 8 
    return df, color_map


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
            
        # Construct Thumbnail if missing
        thumb = data.get("thumbnail")
        if not thumb:
            thumb = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"

        rows.append({
            "id": vid_id,
            "title": data.get("title", "Untitled"),
            "orchestra": orchestra or "Unknown",
            "event": event or "Unknown",
            "year": (data.get("event") or {}).get("year"),
            "videographer": data.get("videographer", "Unknown"),
            "dancers": ", ".join(dancers_list),
            "url": data.get("url"),
            "thumbnail": thumb,
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

# --- Modal: Watch Video (Performance) ---
@st.dialog("Now Playing", width="large")
def watch_video(video_url, title):
    st.subheader(title)
    st.video(video_url)

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
tab_atlas, tab_library, tab_dashboard = st.tabs(["Style Atlas", "Video Library", "Dashboard"])

# --- TAB 1: STYLE ATLAS ---
with tab_atlas:
    # Hardcoded parameters based on user preference
    df, color_map = get_atlas_data(
        dancers, 
        layout_neighbors=15, 
        cluster_neighbors=15, 
        min_dist=1.0, 
        spread=5.0, 
        init_mode='random', 
        decoupling=False
    )
    
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
        
        # Sort so selected is on top
        df = df.sort_values(by='status', ascending=True)

        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            text='name', 
            size='final_size', 
            color='cluster', 
            color_discrete_map=color_map,
            hover_data=['name', 'videos'],
            # Fixed height to fit standard laptop screens without scrolling
            height=550,
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
                
                # Dynamic Zoom: Viewport = 15% of total data range
                # We calculate the total spread of the data first
                total_x = df['x'].max() - df['x'].min()
                total_y = df['y'].max() - df['y'].min()
                
                # The 'radius' (delta) is half of the desired viewport (7.5%)
                delta_x = total_x * 0.075
                delta_y = total_y * 0.075
                
                # Ensure a minimum sensible zoom if distinct points are few
                delta_x = max(delta_x, 5.0)
                delta_y = max(delta_y, 5.0)

                x_range = [tx - delta_x, tx + delta_x]
                y_range = [ty - delta_y, ty + delta_y]

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
            config={'scrollZoom': True, 'displayModeBar': False}
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
            
        # --- Pagination Control ---
        # Initialize limit in session state if not present
        if "lib_limit" not in st.session_state:
            st.session_state.lib_limit = 200
            
        # Apply Limit
        display_df = filtered_df.head(st.session_state.lib_limit)
        
        st.markdown(f"**Showing {len(display_df)} of {len(filtered_df)} matching videos**")
        
        # --- Grid Layout ---
        # We use columns to create a grid
        cols = st.columns(4) # Tighter grid for thumbnails
        for idx, row in display_df.iterrows():
            with cols[idx % 4]:
                with st.container(border=True):
                    # Thumbnail Image
                    st.image(row['thumbnail'], use_container_width=True)
                    
                    # Play Button (Triggers Modal)
                    if st.button(f"▶️ Play", key=f"lib_play_{row['id']}", use_container_width=True):
                        watch_video(row['url'], row['title'])

                    st.caption(f"**{row['title'][:50]}...**")
                    st.caption(f"{row['orchestra']} • {row['event']}")

        # --- Load More Button ---
        if len(display_df) < len(filtered_df):
            if st.button("Load More Videos", use_container_width=True):
                st.session_state.lib_limit += 200
                st.rerun()

# --- TAB 3: DASHBOARD ---
with tab_dashboard:
    st.header("Global Statistics")
    
    # Calculate Population Estimators (Chao1)
    sightings = queue_state.get("video_sightings", {})
    # Fallback for legacy format
    if not sightings and "seen_videos" in queue_state:
        sightings = {v: 1 for v in queue_state["seen_videos"]}
         
    seen_count = len(sightings)
    f1 = sum(1 for c in sightings.values() if c == 1) # Singletons
    f2 = sum(1 for c in sightings.values() if c == 2) # Doubletons
    
    if f2 > 0:
        est_total = seen_count + (f1 ** 2) / (2 * f2)
    else:
        est_total = seen_count + (f1 * (f1 - 1)) / (2 * (f2 + 1))
        
    coverage_pct = (seen_count / est_total * 100) if est_total > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Collected Videos", len(videos), help="Videos successfully extracted and stored in the graph.")
    c2.metric("Total Dancers", len(dancers), help="Unique dancers identified in the graph.")
    c3.metric("Est. Total Videos", f"{int(est_total):,}", help="Chao1 estimate of total discoverable videos based on recapture rate.")
    c4.metric("Coverage", f"{coverage_pct:.1f}%", help="Percentage of the estimated total that has been processed.")
    
    # Queue Stats
    q_len = len(queue_state.get("queue", []))
    c5.metric("Pending Queries", f"{q_len:,}")
    
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
            fig.update_xaxes(categoryorder='total descending', tickangle=-90)
            fig.update_layout(
                yaxis=dict(title="Video Count", fixedrange=True), 
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
        
    with col_queue:
        st.subheader("Discovery Queue")
        queue_list = queue_state.get("queue", [])
        if queue_list:
            # The queue is now a list of [Priority, Query] tuples
            try:
                df_queue = pd.DataFrame(queue_list, columns=["Priority", "Query"])
                # Sort so high priority (low number) is at top
                df_queue = df_queue.sort_values("Priority", ascending=True)
                st.dataframe(df_queue, height=400, hide_index=True)
            except ValueError:
                # Fallback for legacy/migration state if mixed
                st.write("Queue format updating...")
                st.json(queue_list[:10])
        else:
            st.info("Queue is empty.")