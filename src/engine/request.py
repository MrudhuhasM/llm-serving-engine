from dataclasses import dataclass, field
from datetime import datetime
import uuid
@dataclass
class Request:
    prompt: str 
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = field(default=100)
    generated_token_ids: list[int] = field(default_factory=list)
    finished: bool = field(default=False)
    finished_reason: str | None = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)