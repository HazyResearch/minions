from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation between user and assistant."""
    query: str
    local_output: str
    remote_output: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "local_output": self.local_output,
            "remote_output": self.remote_output,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create a ConversationTurn from a dictionary."""
        timestamp = None
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass
                
        return cls(
            query=data.get("query", ""),
            local_output=data.get("local_output", ""),
            remote_output=data.get("remote_output", ""),
            timestamp=timestamp
        )


class ConversationHistory:
    """Manages a history of conversation turns using a sliding window approach with summarization."""
    
    def __init__(self, max_turns: int = 10, turns_per_summary: int = 5, summarize_older_turns: bool = True):
        """
        Initialize a conversation history with a maximum number of turns.
        
        Args:
            max_turns: Maximum number of turns to store in history
            turns_per_summary: Number of turns after which to generate a summary
            summarize_older_turns: Whether to summarize older turns
        """
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        self.summary: str = ""
        self.turns_since_last_summary: int = 0
        self.turns_per_summary = turns_per_summary
        self.summarize_older_turns = summarize_older_turns
    
    def add_turn(self, turn: ConversationTurn, remote_client=None) -> None:
        """
        Add a turn to the conversation history, removing oldest if at capacity.
        If summarize_older_turns is enabled, will periodically summarize older turns.
        
        Args:
            turn: The conversation turn to add
            remote_client: The remote client to use for generating summaries (if None, no summarization will be done)
        """
        self.turns.append(turn)
        self.turns_since_last_summary += 1
        
        # If we have exceeded the max turns and summarization is enabled
        if len(self.turns) > self.max_turns and self.summarize_older_turns and remote_client:
            # If it's time to create a new summary
            if self.turns_since_last_summary >= self.turns_per_summary:
                # Get the oldest turns that exceed our max
                turns_to_summarize = self.turns[:(len(self.turns) - self.max_turns + 1)]
                # Remove them from the current turns list
                self.turns = self.turns[(len(self.turns) - self.max_turns + 1):]
                # Generate a summary of these turns and older summary
                self._summarize_turns(turns_to_summarize, remote_client)
                # Reset counter
                self.turns_since_last_summary = 0
            else:
                # Just remove the oldest turn if we're not summarizing yet
                self.turns.pop(0)
        # If summarization is not enabled, just use the sliding window approach
        elif len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def _summarize_turns(self, turns_to_summarize: List[ConversationTurn], remote_client) -> None:
        """
        Generate a summary of the given turns and combine with existing summary.
        
        Args:
            turns_to_summarize: The conversation turns to summarize
            remote_client: The remote client to use for generating summaries
        """
        from minions.prompts.multi_turn import CONVERSATION_SUMMARY_PROMPT, MULTI_TURN_CONVERSATION_HISTORY_FORMAT
        
        # Format the turns to summarize
        formatted_history = ""
        for i, turn in enumerate(turns_to_summarize):
            formatted_history += MULTI_TURN_CONVERSATION_HISTORY_FORMAT.format(
                query_index=i+1,
                query=turn.query,
                response=turn.remote_output
            )
        
        # If we already have a summary, include it in what we're going to summarize
        if self.summary:
            formatted_history = f"Previous Summary:\n{self.summary}\n\nNew Conversation Turns:\n{formatted_history}"
        
        # Create the prompt for summarization
        prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_history=formatted_history)
        
        # Generate the summary using the remote client
        try:
            messages = [{"role": "user", "content": prompt}]
            outputs, _ = remote_client.chat(messages)
            new_summary = outputs[0]
            self.summary = new_summary
        except Exception as e:
            # In case of errors with the remote client, keep the old summary
            print(f"Error generating conversation summary: {e}")
    
    def get_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get the most recent n turns.
        
        Args:
            n: Number of recent turns to retrieve, or None for all turns
            
        Returns:
            List of conversation turns, most recent last
        """
        if n is None or n >= len(self.turns):
            return self.turns
        return self.turns[-n:]
    
    def get_latest_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent turn or None if history is empty."""
        if not self.turns:
            return None
        return self.turns[-1]
    
    def clear(self) -> None:
        """Clear all turns from the history."""
        self.turns = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_turns": self.max_turns,
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": self.summary,
            "turns_since_last_summary": self.turns_since_last_summary,
            "turns_per_summary": self.turns_per_summary,
            "summarize_older_turns": self.summarize_older_turns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """Create a ConversationHistory from a dictionary."""
        history = cls(
            max_turns=data.get("max_turns", 10),
            turns_per_summary=data.get("turns_per_summary", 5),
            summarize_older_turns=data.get("summarize_older_turns", True)
        )
        history.summary = data.get("summary", "")
        history.turns_since_last_summary = data.get("turns_since_last_summary", 0)
        
        for turn_data in data.get("turns", []):
            turn = ConversationTurn.from_dict(turn_data)
            history.turns.append(turn)
        return history 