from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Dancer(BaseModel):
    name: str = Field(..., description="The name of the dancer. Prefer full names if available (e.g. 'Mariano Chicho Frumboli' instead of 'Chicho').")
    role: Optional[Literal["leader", "follower"]] = Field(None, description="The role of the dancer if explicitly stated or clearly inferable.")

class Music(BaseModel):
    orchestra: Optional[str] = Field(None, description="The orchestra, band, or artist performing the music (e.g., 'Juan D\'Arienzo', 'Osvaldo Pugliese', 'Color Tango').")
    song: Optional[str] = Field(None, description="The title of the song.")
    singer: Optional[str] = Field(None, description="The singer (cantor) if mentioned (e.g. 'Alberto Echagüe').")
    style: Optional[Literal["tango", "milonga", "vals", "alternative", "electronic"]] = Field(None, description="The style of the music.")

class Event(BaseModel):
    name: Optional[str] = Field(None, description="The name of the festival, marathon, or organizer (e.g. 'CITA', 'Tango Element', 'Torino Tango Festival').")
    year: Optional[int] = Field(None, description="The year the event took place.")
    location: Optional[str] = Field(None, description="The city or country.")

class VideoContent(BaseModel):
    """
    Structured extraction of entities from a Tango dance video metadata.
    """
    dancers: List[Dancer] = Field(default_factory=list, description="List of dancers performing.")
    music: Optional[Music] = Field(None, description="Music details.")
    event: Optional[Event] = Field(None, description="Event details.")
    videographer: Optional[str] = Field(None, description="The name of the videographer or filming channel (e.g. '030tango', 'Gancho', 'Focal Tango').")
    video_type: Literal["performance", "social", "class", "talk", "other"] = Field("performance", description="The context of the video. 'performance' usually implies a stage or cleared floor with an audience.")
    
    suggested_search_queries: List[str] = Field(
        default_factory=list, 
        description="3-5 new search queries derived from this content to find similar videos. Use specific entity names combined with years or festivals (e.g. 'Chicho Frumboli 2024', 'Torino Tango Festival performance')."
    )