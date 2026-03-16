import streamlit as st
import json
import os
import random
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px

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
        /* Allow pinch-to-zoom on Plotly charts instead of browser handling it */
        .stPlotlyChart,
        .stPlotlyChart > div,
        .stPlotlyChart > div > div,
        .stPlotlyChart iframe {
            touch-action: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_resource(ttl=3600)
def load_data():
    db_path = "graph_db.json"
    if not os.path.exists(db_path):
        return {"videos": {}, "dancers": {}, "stats": {}}
    
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        return {"videos": {}, "dancers": {}, "stats": {}}

@st.cache_data(ttl=5)
def load_queue_state():
    state_path = "queue_state.json"
    if not os.path.exists(state_path):
        return {"queue": [], "video_sightings": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"queue": [], "video_sightings": []}

# --- Atlas Generators ---

def _flatten_atlas_groups(grouped_points, color_map):
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
                "videos": primary["videos"],
                "cluster": primary["cluster"],
                "members": names
            })
    
    if not plot_data: return pd.DataFrame(), {}
    
    df = pd.DataFrame(plot_data)
    df['size_log'] = np.log1p(df['videos']) * 8 
    if '-1' not in color_map: color_map['-1'] = '#7f7f7f'
    
    return df, color_map

@st.cache_data
def get_static_atlas(_dancers_data, _version_id=None):
    """
    Fast path: Reads pre-computed UMAP coordinates from graph_db.json.
    """
    grouped_points = {}
    color_map = {}

    for name, info in _dancers_data.items():
        embedding = info.get("style_embedding")
        if not embedding or "x" not in embedding:
            continue

        if "cluster" in embedding and "color" in embedding:
            color_map[embedding["cluster"]] = embedding["color"]

        key = (round(embedding["x"], 4), round(embedding["y"], 4))
        if key not in grouped_points:
            grouped_points[key] = []

        grouped_points[key].append({
            "name": name,
            "videos": len(info.get("videos", [])),
            "x": embedding["x"],
            "y": embedding["y"],
            "cluster": embedding.get("cluster", "0")
        })

    return _flatten_atlas_groups(grouped_points, color_map)

@st.cache_data
def get_library_data(_videos_data, _version_id=None):
    """
    Flattens video dictionary into a DataFrame for the Library tab.
    """
    if not _videos_data:
        return pd.DataFrame()

    rows = []
    for vid_id, data in _videos_data.items():
        orchestra = None
        music_data = data.get("music") or {}
        if music_data.get("orchestra"):
            orchestra = music_data["orchestra"]
            
        event = None
        event_data = data.get("event") or {}
        if event_data.get("name"):
            event = event_data["name"]
            
        dancers_list = []
        if data.get("performances"):
            for p in data["performances"]:
                dancers_list.extend([d.get("name") for d in p.get("dancers", [])])
        elif data.get("dancers"):
            dancers_list = [d.get("name") for d in data["dancers"]]
            
        thumb = data.get("thumbnail")
        if not thumb:
            thumb = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"

        rows.append({
            "id": vid_id,
            "title": data.get("title", "Untitled"),
            "orchestra": orchestra or "Unknown",
            "event": event or "Unknown",
            "year": event_data.get("year"),
            "videographer": data.get("videographer", "Unknown"),
            "dancers": ", ".join(dancers_list),
            "url": data.get("url"),
            "thumbnail": thumb,
            "duration": data.get("duration", 0)
        })
    
    return pd.DataFrame(rows)


data = load_data()
videos = data.get("videos", {})
dancers = data.get("dancers", {})
stats = data.get("stats", {})
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
    
    selected = st.selectbox(
        "Type to search...", 
        options=all_names, 
        index=None,
        key="search_modal_box",
        placeholder="Name..."
    )

    if st.button("Locate", type="primary", use_container_width=True):
        if selected:
            st.session_state.selected_dancer = selected
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
        
        c_stats, c_related = st.columns([1, 1], gap="large")
        with c_stats:
            st.header(dancer_name)
            m1, m2, m3 = st.columns(3)
            m1.metric("Videos", len(info.get("videos", [])))
            m2.metric("Events", len(info.get("events", {})))
            m3.metric("Partners", len(info.get("partners", {})))

        with c_related:
            st.subheader("Related Dancers")
            similar = info.get("similar_dancers", [])
            if similar:
                sim_cols = st.columns(2)
                for idx, item in enumerate(similar[:6]): 
                    name = item.get("name")
                    score = item.get("score", 0)
                    with sim_cols[idx % 2]:
                        if st.button(f"{name} ({score:.2f})", key=f"modal_sim_{name}", use_container_width=True):
                            st.session_state.selected_dancer = name
                            # Reset playing videos when switching context
                            st.session_state.playing_videos = set()
                            st.rerun()
            else:
                st.caption("No similar dancers found.")

        st.divider()
        st.subheader("Performance History")
        video_ids = info.get("videos", [])

        # Initialize playing state for this session
        if "playing_videos" not in st.session_state:
            st.session_state.playing_videos = set()

        v_cols = st.columns(2)
        for i, vid_id in enumerate(video_ids):
            v = videos.get(vid_id)
            if not v: continue
            with v_cols[i % 2]:
                with st.container(border=True):
                    # Optimized Loading: Thumbnail first, Video on demand
                    if vid_id in st.session_state.playing_videos:
                        if v.get("url"):
                            st.video(v["url"])
                    else:
                        thumb = v.get("thumbnail") or f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"
                        st.image(thumb, use_container_width=True)
                        if st.button("▶️ Play", key=f"profile_play_{vid_id}", use_container_width=True):
                            st.session_state.playing_videos.add(vid_id)
                            st.rerun()

                    event_data = v.get('event') or {}
                    event_name = event_data.get('name', 'Unknown Event')
                    title = v.get('title', 'Untitled')
                    st.markdown(f"**{title}**")
                    st.caption(f"📍 {event_name} • ⏱️ {format_duration(v.get('duration', 0))}")

# --- Main Layout ---
st.markdown("### 🇦🇷 TangoGraph")

# Top-level Tabs
tab_atlas, tab_library, tab_dashboard, tab_random = st.tabs(["Style Atlas", "Video Library", "Dashboard", "Random"])

# --- TAB 1: STYLE ATLAS ---
with tab_atlas:
    df, color_map = get_static_atlas(dancers, id(dancers))
    
    if df.empty:
        st.warning("No style embeddings found. Please wait for the backend maintenance cycle to populate them.")
    else:
        st.caption("A map of the tango world based on youtube performance history. Dancers who attend the same events or perform to similar music appear closer together.")
        # --- Controls Row ---
        c_search, c_info = st.columns([1, 5])
        with c_search:
            if st.button("🔍 Search", use_container_width=True):
                all_names = sorted(df['name'].tolist())
                show_search_modal(all_names)
        with c_info:
            active_dancer = st.session_state.get("selected_dancer", None)
            if active_dancer:
                col_i1, col_i2 = st.columns([5, 1])
                with col_i1:
                    st.info(f"Selected: **{active_dancer}**", icon="📍")
                with col_i2:
                    if st.button("Profile", key="btn_open_profile", use_container_width=True):
                        show_dancer_details(active_dancer)

        # --- Zoom Controls ---
        if "atlas_zoom" not in st.session_state:
            st.session_state.atlas_zoom = 1.0

        c_zin, c_zout, c_zreset, _ = st.columns([1, 1, 1, 9])
        with c_zin:
            if st.button("➕", key="zoom_in", use_container_width=True):
                st.session_state.atlas_zoom *= 0.6
                st.rerun()
        with c_zout:
            if st.button("➖", key="zoom_out", use_container_width=True):
                st.session_state.atlas_zoom /= 0.6
                st.rerun()
        with c_zreset:
            if st.button("↩️", key="zoom_reset", use_container_width=True):
                st.session_state.atlas_zoom = 1.0
                st.rerun()
        # --- Plot ---
        df['status'] = df['name'].apply(lambda x: 'Selected' if x == active_dancer else 'Normal')
        df['final_size'] = df.apply(lambda row: 30 if row['name'] == active_dancer else row['size_log'], axis=1)
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
            height=550,
            custom_data=['name'],
            render_mode='webgl' # Optimization: 10x faster rendering
        )
        
        fig.update_traces(
            textposition='top center', 
            textfont=dict(size=10, color='rgba(200,200,200,0.9)'),
            marker=dict(opacity=0.8, line=dict(width=0))
        )
        
        if active_dancer:
            selected_row = df[df['name'] == active_dancer]
            if not selected_row.empty:
                fig.add_scatter(
                    x=selected_row['x'],
                    y=selected_row['y'],
                    mode='markers',
                    marker=dict(
                        size=30,
                        color='rgba(0,0,0,0)',
                        line=dict(width=4, color='Red'),
                        symbol='circle'
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    customdata=selected_row[['name']]
                )

        # Auto-Centering
        x_range = None
        y_range = None
        zoom = st.session_state.atlas_zoom

        if active_dancer:
            target_row = df[df['name'] == active_dancer]
            if not target_row.empty:
                tx = target_row.iloc[0]['x']
                ty = target_row.iloc[0]['y']
                # Dynamic zoom based on global scale
                total_x = df['x'].max() - df['x'].min()
                total_y = df['y'].max() - df['y'].min()
                delta_x = max(total_x * 0.05, 1.5) * zoom
                delta_y = max(total_y * 0.05, 1.5) * zoom

                x_range = [tx - delta_x, tx + delta_x]
                y_range = [ty - delta_y, ty + delta_y]
        else:
            # Default view: Center on "Center of Mass" (Mean) rather than bounding box center
            if not df.empty:
                centroid_x = df['x'].mean()
                centroid_y = df['y'].mean()

                # Use std dev to determine zoom level, excluding extreme outliers
                # 2.5 std devs covers ~98% of data in normal dist, good for showing the main cluster
                std_x = df['x'].std()
                std_y = df['y'].std()

                # Fallback span if std is tiny
                span_x = ((std_x * 2.5) if std_x > 0.5 else 5.0) * zoom
                span_y = ((std_y * 2.5) if std_y > 0.5 else 5.0) * zoom

                x_range = [centroid_x - span_x, centroid_x + span_x]
                y_range = [centroid_y - span_y, centroid_y + span_y]

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

        selection = st.plotly_chart(
            fig, 
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"atlas_chart_{active_dancer}", 
            config={'scrollZoom': True, 'displayModeBar': False}
        )

        if selection and selection.get("selection") and len(selection["selection"]["points"]) > 0:
            clicked_name = selection["selection"]["points"][0]["customdata"][0]
            # Prevent "Ghost Modal": Only open if selection actually changed from current state.
            if clicked_name != st.session_state.get("selected_dancer"):
                st.session_state.selected_dancer = clicked_name
                st.session_state.show_modal = True
                st.rerun()
        
        # Handle modal display
        if st.session_state.get("show_modal", False) and active_dancer:
            show_dancer_details(active_dancer)
            # Reset modal state to prevent infinite loops if logic requires,
            # but st.dialog handles persistence until dismissed.
            # We don't manually set False here; closing the dialog triggers a rerun where we won't enter this block if dismissed.

# --- TAB 2: VIDEO LIBRARY ---
with tab_library:
    df_videos = get_library_data(videos, id(videos))
    if df_videos.empty:
        st.info("No videos found yet.")
    else:
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        with c_filter1:
            all_orchestras = sorted([x for x in df_videos['orchestra'].unique() if x != "Unknown"])
            sel_orch = st.multiselect("Orchestra", options=all_orchestras)
        with c_filter2:
            all_events = sorted([x for x in df_videos['event'].unique() if x != "Unknown"])
            sel_event = st.multiselect("Event", options=all_events)
        with c_filter3:
            search_dancer = st.text_input("Dancer Name", placeholder="e.g. Chicho")

        filtered_df = df_videos.copy()
        if sel_orch:
            filtered_df = filtered_df[filtered_df['orchestra'].isin(sel_orch)]
        if sel_event:
            filtered_df = filtered_df[filtered_df['event'].isin(sel_event)]
        if search_dancer:
            filtered_df = filtered_df[filtered_df['dancers'].str.contains(search_dancer, case=False, na=False)]
            
        if "lib_limit" not in st.session_state:
            st.session_state.lib_limit = 200
            
        display_df = filtered_df.head(st.session_state.lib_limit)
        st.markdown(f"**Showing {len(display_df)} of {len(filtered_df)} matching videos**")
        
        cols = st.columns(4)
        for idx, row in display_df.iterrows():
            with cols[idx % 4]:
                with st.container(border=True):
                    st.image(row['thumbnail'], use_container_width=True)
                    if st.button(f"▶️ Play", key=f"lib_play_{row['id']}", use_container_width=True):
                        watch_video(row['url'], row['title'])
                    st.caption(f"**{row['title'][:50]}...**")
                    st.caption(f"{row['orchestra']} • {row['event']}")

        if len(display_df) < len(filtered_df):
            if st.button("Load More Videos", use_container_width=True):
                st.session_state.lib_limit += 200
                st.rerun()

# --- TAB 3: DASHBOARD ---
with tab_dashboard:
    st.header("Global Statistics")
    sightings = queue_state.get("video_sightings", {})
    if not sightings and "seen_videos" in queue_state:
        sightings = {v: 1 for v in queue_state["seen_videos"]}
    seen_count = len(sightings)
    f1 = sum(1 for c in sightings.values() if c == 1)
    f2 = sum(1 for c in sightings.values() if c == 2)
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
    c5.metric("Pending Queries", f"{len(queue_state.get('queue', [])):,}")
    
    st.divider()
    col_charts, col_queue = st.columns([2, 1])
    
    with col_charts:
        def plot_sorted_bar(data_list, title, color):
            if not data_list: return
            counts = Counter(data_list)
            top_items = counts.most_common(40)
            if not top_items: return
            df_counts = pd.DataFrame(top_items, columns=['Name', 'Count'])
            fig = px.bar(df_counts, x='Name', y='Count', title=title, color_discrete_sequence=[color])
            fig.update_xaxes(categoryorder='total descending', tickangle=-90)
            fig.update_layout(yaxis=dict(title="Video Count", fixedrange=True), xaxis=dict(fixedrange=False), dragmode='pan')
            st.plotly_chart(fig, use_container_width=True)

        if stats.get("top_events"):
            plot_sorted_bar(stats["top_events"], "Top Festivals (Pareto)", "#EF553B")
        else:
            all_events = []
            for v in videos.values():
                if v.get("event") and v.get("event", {}).get("name"):
                    all_events.append(v["event"]["name"])
            plot_sorted_bar(all_events, "Top Festivals (Pareto)", "#EF553B")

        if stats.get("top_videographers"):
            plot_sorted_bar(stats["top_videographers"], "Top Videographers", "#00CC96")
        else:
            all_vids = []
            for v in videos.values():
                if v.get("videographer"):
                    all_vids.append(v["videographer"])
            plot_sorted_bar(all_vids, "Top Videographers", "#00CC96")

        if stats.get("top_orchestras"):
            plot_sorted_bar(stats["top_orchestras"], "Top Orchestras", "#AB63FA")
        else:
            all_orch = []
            unique_tags = set()
            for v in videos.values():
                if v.get("music") and v["music"].get("orchestra"):
                    all_orch.append(v["music"]["orchestra"])
                if v.get("tags"):
                    unique_tags.update(v["tags"])
            st.divider()
            m_orch, m_tags = st.columns(2)
            m_orch.metric("Unique Orchestras", f"{len(set(all_orch)):,}")
            m_tags.metric("Unique Tags", f"{len(unique_tags):,}")
            plot_sorted_bar(all_orch, "Top Orchestras", "#AB63FA")
        
    with col_queue:
        st.subheader("Discovery Queue")
        queue_list = queue_state.get("queue", [])
        if queue_list:
            try:
                df_queue = pd.DataFrame(queue_list, columns=["Priority", "Query"])
                df_queue = df_queue.sort_values("Priority", ascending=True)
                st.dataframe(df_queue, height=400, hide_index=True)
            except ValueError:
                st.write("Queue format updating...")
                st.json(queue_list[:10])
        else:
            st.info("Queue is empty.")
# --- TAB 4: RANDOM ---
with tab_random:
    # Filter for videos that actually have identified dancers
    candidate_ids = [
        vid_id for vid_id, data in videos.items()
        if (data.get("performances") and any(p.get("dancers") for p in data["performances"]))
        or data.get("dancers")
    ]

    if not candidate_ids:
        st.info("No videos with identified dancers found.")
    else:
        c_rand_vid, c_rand_info = st.columns([2, 1])

        # Initialize session state for random video if not present
        if "random_video_id" not in st.session_state:
            st.session_state["random_video_id"] = random.choice(candidate_ids)

        with c_rand_info:
            st.markdown("### Serendipity")
            if st.button("🎲 Shuffle Video", type="primary", use_container_width=True):
                st.session_state["random_video_id"] = random.choice(candidate_ids)

            vid_id = st.session_state["random_video_id"]
            vid_data = videos.get(vid_id, {})

            st.divider()
            st.markdown(f"**{vid_data.get('title', 'Untitled')}**")

            # Extract Metadata
            dancers_list = []
            if vid_data.get("performances"):
                for p in vid_data["performances"]:
                    dancers_list.extend([d.get("name") for d in p.get("dancers", [])])

            st.markdown(f"**💃 Dancers:** {', '.join(dancers_list) or 'Unknown'}")

            # Safe Access for Nested Objects
            event_name = (vid_data.get('event') or {}).get('name', 'Unknown')
            st.markdown(f"**📍 Event:** {event_name}")

            music_data = vid_data.get('music') or {}
            orch_name = music_data.get('orchestra', 'Unknown')
            st.markdown(f"**🎻 Music:** {orch_name}")

            tags = vid_data.get("tags", [])
            if tags:
                st.caption(f"Tags: {', '.join(tags[:5])}")

        with c_rand_vid:
            if vid_id:
                url = vid_data.get("url", f"https://www.youtube.com/watch?v={vid_id}")
                st.video(url)