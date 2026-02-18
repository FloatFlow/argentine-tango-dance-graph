import yt_dlp
from typing import List, Dict, Optional, Any
from loguru import logger
import gin

@gin.configurable
class TubeClient:
    def __init__(self, proxy_url: Optional[str] = None):
        self.proxy_url = proxy_url
        # Base options for fast discovery
        self.base_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'skip_download': True,
            'extract_flat': True, # Only get feed metadata (fast), no page visit
        }
        if self.proxy_url:
            self.base_opts['proxy'] = self.proxy_url

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a YouTube search and returns a list of video summaries.
        These summaries usually contain 'id', 'title', 'uploader', but not full descriptions.
        """
        search_query = f"ytsearch{limit}:{query}"
        logger.info(f"Searching YouTube: {search_query}")
        
        with yt_dlp.YoutubeDL(self.base_opts) as ydl:
            try:
                result = ydl.extract_info(search_query, download=False)
                if result and 'entries' in result:
                    # Filter out Nones which happen on errors
                    return [e for e in result['entries'] if e]
                return []
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                return []

    def get_channel_videos(self, channel_url: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetches recent videos from a specific channel URL.
        """
        logger.info(f"Scraping channel: {channel_url}")
        opts = self.base_opts.copy()
        opts['playlistend'] = limit
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(channel_url, download=False)
                if result and 'entries' in result:
                    return [e for e in result['entries'] if e]
                return []
            except Exception as e:
                logger.error(f"Failed to fetch channel {channel_url}: {e}")
                return []

    def fetch_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the FULL metadata for a single video, including description and tags.
        This is slower as it visits the video page.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        # We disable extract_flat to get the full page data
        opts = self.base_opts.copy()
        opts['extract_flat'] = False
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                return ydl.extract_info(url, download=False)
            except Exception as e:
                logger.error(f"Failed to fetch details for video {video_id}: {e}")
                return None