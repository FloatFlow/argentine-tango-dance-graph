import streamlit as st
import json
import os
from collections import Counter
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="Tango Graph Explorer",
    page_icon="💃",
    layout="wide"
)

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

def get_similar_dancers(target_name, dancers_db):
    """Re-implements the Jaccard logic for the frontend"""
    if target_name not in dancers_db:
        return []
    
    target_info = dancers_db[target_name]
    # Handle case where partners might be empty or old format
    target_partners = set(target_info.get("partners", {}).keys())
    
    scores = []
    
    for other_name, info in dancers_db.items():
        if other_name == target_name:
            continue
        
        other_partners = set(info.get("partners", {}).keys())
        
        intersection = len(target_partners.intersection(other_partners))
        union = len(target_partners.union(other_partners))
        
        if union > 0:
            score = intersection / union
            if score > 0.1:
                scores.append((other_name, score))
    
    # Sort by score descending
    return sorted(scores, key=lambda x: x[1], reverse=True)[:5]

# --- Sidebar ---
st.sidebar.title="🇦🇷 Tango Explorer"
st.sidebar.markdown("Browse the semantic graph of Argentine Tango videos.")

mode = st.sidebar.radio("Mode", ["Dashboard", "Dancer Explorer", "Style Atlas", "Video Browser"])

if st.sidebar.button("Refresh Data"):
    load_data.clear()
    st.rerun()

# --- Main Content ---
if mode == "Dashboard":
    st.title("Graph Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Videos", len(videos))
    col2.metric("Total Dancers", len(dancers))
    
    # Top Events
    events = []
    for v in videos.values():
        if v.get("event") and v["event"].get("name"):
            events.append(v["event"]["name"])
    
    st.subheader("Top Festivals")
    if events:
        top_events = Counter(events).most_common(15)
        st.bar_chart(dict(top_events))
    else:
        st.info("No event data collected yet.")

elif mode == "Dancer Explorer":
    st.title("Dancer Explorer")
    
    all_dancers = sorted(list(dancers.keys()))
    if not all_dancers:
        st.warning("No dancers found in database yet.")
        st.stop()
    
    # Initialize session state for navigation
    if "selected_dancer" not in st.session_state:
        st.session_state.selected_dancer = all_dancers[0]

    # Handle case where session state dancer might disappear on reload (rare)
    if st.session_state.selected_dancer not in all_dancers:
        st.session_state.selected_dancer = all_dancers[0]
        
    # Callback to sync selectbox with session state
    def update_dancer():
        st.session_state.selected_dancer = st.session_state.dancer_select

    # The Selectbox
    selected_dancer = st.selectbox(
        "Select a Dancer", 
        all_dancers, 
        key="dancer_select",
        index=all_dancers.index(st.session_state.selected_dancer),
        on_change=update_dancer
    )
    
    if selected_dancer:
        info = dancers[selected_dancer]
        
        # Stats Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Videos", len(info.get("videos", [])))
        c1.metric("Unique Partners", len(info.get("partners", {})))
        
        # Similar Dancers
        st.subheader("Similar Style / Network")
        similar = get_similar_dancers(selected_dancer, dancers)
        if similar:
            cols = st.columns(5)
            for idx, (name, score) in enumerate(similar):
                if idx < 5:
                    cols[idx].caption(f"Match: {score:.2f}")
                    if cols[idx].button(name, key=f"sim_{name}"):
                        # Update state and rerun to 'jump' to the new dancer
                        st.session_state.selected_dancer = name
                        st.rerun()
        else:
            st.info("Not enough overlap data to find similar dancers yet.")

        st.divider()

        # Video Grid
        st.subheader(f"Videos featuring {selected_dancer}")
        video_ids = info.get("videos", [])
        
        # Display in rows of 3
        cols = st.columns(3)
        for i, vid_id in enumerate(video_ids):
            vid_data = videos.get(vid_id)
            if not vid_data:
                continue
                
            with cols[i % 3]:
                # Thumbnail & Link
                thumb = vid_data.get("thumbnail")
                title = vid_data.get("title", "Unknown Title")
                url = vid_data.get("url", "#")
                duration = format_duration(vid_data.get("duration", 0))
                year = vid_data.get('event', {}).get('year', '????')
                event_name = vid_data.get('event', {}).get('name', 'Unknown Event')
                
                if thumb:
                    st.image(thumb, use_container_width=True)
                
                st.markdown(f"**[{title}]({url})**")
                st.caption(f"⏱️ {duration} • 📅 {year}")
                st.caption(f"📍 {event_name}")
                
                # Show partners in this video
                content_dancers = vid_data.get('dancers', [])
                # Handle Pydantic dump or raw dict
                dancer_names = [d['name'] if isinstance(d, dict) else d.name for d in content_dancers]
                
                others = [d for d in dancer_names if d != selected_dancer]
                if others:
                    st.text(f"with: {', '.join(others)}")
                st.divider()

elif mode == "Style Atlas":
    st.title("Style Atlas (Embedding Space)")
    st.markdown("This map projects dancers into 2D space based on their partner network. **Dancers who are close together likely share the same social or stylistic circle.**")
    
    # Extract embeddings from DB
    plot_data = []
    
    for name, info in dancers.items():
        embedding = info.get("style_embedding")
        if embedding:
            plot_data.append({
                "name": name,
                "x": embedding["x"],
                "y": embedding["y"],
                "videos": len(info.get("videos", []))
            })
            
    if not plot_data:
        st.warning("No style embeddings calculated yet. Wait for the maintenance cycle to run (requires > 5 active dancers).")
    else:
        df = pd.DataFrame(plot_data)
        
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            text='name', 
            size='videos', 
            hover_data=['name'],
            title="Semantic Proximity of Dancers (t-SNE)"
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

elif mode == "Video Browser":
    st.title("Latest Videos")
    
    if not videos:
        st.warning("No videos found.")
    else:
        # Just show latest 20 added
        latest_ids = list(videos.keys())[-20:]
        
        for vid_id in reversed(latest_ids):
            v = videos[vid_id]
            with st.expander(f"{v.get('title', 'Untitled')} | {v.get('event', {}).get('name', 'Unknown Event')}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    if v.get("thumbnail"):
                        st.image(v["thumbnail"])
                    st.caption(f"Duration: {format_duration(v.get('duration', 0))}")
                with c2:
                    st.markdown(f"[Watch on YouTube]({v.get('url')})")
                    st.write(v.get("description", "")[:300] + "...")
                    
                    st.write("**Dancers:**")
                    # extract names safely
                    dancers_list = v.get("dancers", [])
                    names = [d['name'] if isinstance(d, dict) else d.name for d in dancers_list]
                    st.write(", ".join(names))
                    
                    if v.get("videographer"):
                        st.write(f"**Filmed by:** {v.get('videographer')}")