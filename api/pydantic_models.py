from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, List

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class SourceItem(BaseModel):
    file_id: Optional[int] = None
    file_name: Optional[str] = None
    block_id: Optional[str] = None
    block_type: Optional[str] = None
    page: Optional[int] = None
    asset_base: Optional[str] = None
    image_path: Optional[str] = None
    table_path: Optional[str] = None
    table_csv_path: Optional[str] = None
    original_text_preview: Optional[str] = None
    rank: Optional[int] = None
    cited: bool = False
    text: Optional[str] = None  # full original chunk

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    sources: list[SourceItem] = Field(default_factory=list)

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int