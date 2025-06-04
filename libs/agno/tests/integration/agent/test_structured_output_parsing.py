from typing import Dict, List, Any

import pytest
from pydantic import BaseModel, Field

from agno.agent import Agent, RunResponse

try:
    from agno.models.openai.chat import OpenAIChat
    openai_available = True
except ImportError:
    openai_available = False
    OpenAIChat: Any = None


@pytest.mark.skipif(not openai_available, reason="OpenAI package not installed")
def test_structured_output_parsing_with_quotes():
    class MovieScript(BaseModel):
        script: str = Field(..., description="The script of the movie.")
        name: str = Field(..., description="Give a name to this movie")
        characters: List[str] = Field(..., description="Name of characters for this movie.")

    movie_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You help people write movie scripts. Always add some example dialog in your scripts in double quotes.",
        response_model=MovieScript,
    )

    response: RunResponse = movie_agent.run("New York")
    assert isinstance(response.content, MovieScript)
    assert response.content.script is not None
    assert response.content.name is not None
    assert response.content.characters is not None


@pytest.mark.skipif(not openai_available, reason="OpenAI package not installed")  
class TestJsonQuotesFixIntegration:
    """Integration tests for JSON quotes fix through Agent scenarios."""

    class MovieScript(BaseModel):
        setting: str = Field(description="Movie setting")
        name: str = Field(description="Movie name") 
        characters: List[str] = Field(description="Character names")
        storyline: str = Field(description="Movie storyline")
        dialogue: str = Field(description="Sample dialogue with quotes")
        rating: Dict[str, int] = Field(description="Ratings")

    class ConversationAnalysis(BaseModel):
        speaker_name: str = Field(description="Name of the speaker")
        quoted_text: str = Field(description="Exact quoted text from conversation")
        emotion: str = Field(description="Emotional tone")
        context: str = Field(description="Contextual information")

    class DialogueScript(BaseModel):
        character: str = Field(description="Character name")
        line: str = Field(description="Dialogue line with potential quotes")
        stage_direction: str = Field(description="Stage direction")

    def _create_mock_openai_response(self, content: str):
        """Helper to create properly structured OpenAI response mock."""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.error = None
        mock_response.id = "test-response-id"
        
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = content
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_response.choices = [mock_choice]
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 200
        mock_usage.total_tokens = 300
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200
        mock_usage.cached_tokens = 0
        mock_usage.cache_write_tokens = 0
        
        mock_usage.prompt_tokens_details = {
            "audio_tokens": 0,
            "cached_tokens": 0
        }
        
        mock_usage.completion_tokens_details = {
            "audio_tokens": 0,
            "reasoning_tokens": 0
        }
        
        mock_response.usage = mock_usage
        
        return mock_response

    def test_agent_movie_script_with_dialogue_quotes(self):
        """Test Agent with MovieScript model handling dialogue with quotes."""
        from unittest.mock import patch

        problematic_response = '''```json
{
    "setting": "A dark alley in New York City",
    "name": "The Final Stand",
    "characters": ["Detective Johnson", "The Shadow"],
    "storyline": "A detective must catch a killer who always says 'catch me if you can' before disappearing.",
    "dialogue": "He looked at me and said 'You will never catch me, detective' with a sinister smile.",
    "rating": {"story": 8, "acting": 7}
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You write movie scripts with dialogue.",
            response_model=self.MovieScript,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(problematic_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Write a thriller movie script")
            
            assert isinstance(result.content, self.MovieScript)
            assert result.content.name == 'The Final Stand'
            assert 'Detective Johnson' in result.content.characters
            assert 'The Shadow' in result.content.characters
            assert "'You will never catch me, detective'" in result.content.dialogue
            assert "'catch me if you can'" in result.content.storyline

    def test_agent_conversation_analysis_with_nested_quotes(self):
        """Test Agent with ConversationAnalysis model handling nested quotes."""
        from unittest.mock import patch

        problematic_response = '''```json
{
    "speaker_name": "Sarah Williams",
    "quoted_text": "He told me 'I cannot believe she said no to the proposal' and then walked away.",
    "emotion": "frustrated",
    "context": "During the meeting, she mentioned 'the client specifically said we need better terms' repeatedly."
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You analyze conversations and extract quoted text.",
            response_model=self.ConversationAnalysis,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(problematic_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Analyze this conversation")
            
            assert isinstance(result.content, self.ConversationAnalysis)
            assert result.content.speaker_name == 'Sarah Williams'
            assert "'I cannot believe she said no to the proposal'" in result.content.quoted_text
            assert "'the client specifically said we need better terms'" in result.content.context

    def test_agent_dialogue_script_with_complex_quotes(self):
        """Test Agent with DialogueScript model handling complex stage directions and dialogue."""
        from unittest.mock import patch

        problematic_response = '''```json
{
    "character": "Detective Thompson",
    "line": "The suspect said 'I was at the Blue Moon bar' but the bartender says otherwise.",
    "stage_direction": "Points to the evidence board showing 'ALIBI: Blue Moon Bar - DISPUTED'"
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You write dialogue scripts.",
            response_model=self.DialogueScript,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(problematic_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Write a detective dialogue")
            
            assert isinstance(result.content, self.DialogueScript)
            assert result.content.character == 'Detective Thompson'
            assert "'I was at the Blue Moon bar'" in result.content.line
            assert "'ALIBI: Blue Moon Bar - DISPUTED'" in result.content.stage_direction

    def test_agent_multiline_dialogue_with_quotes(self):
        """Test Agent handles multiline content with quotes correctly."""
        from unittest.mock import patch

        problematic_response = '''```json
{
    "setting": "A spaceship bridge",
    "name": "Star Command Protocol", 
    "characters": ["Captain Sterling"],
    "storyline": "The captain must decide when the AI says 'All systems are go for launch' but sensors show danger.",
    "dialogue": "Captain shouted across the bridge: 'All hands, prepare for emergency protocol!' The computer says 'danger detected' but we have no choice!",
    "rating": {"story": 10, "acting": 9}
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You write sci-fi movie scripts.",
            response_model=self.MovieScript,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(problematic_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Write a sci-fi script")
            
            assert isinstance(result.content, self.MovieScript)
            assert result.content.name == 'Star Command Protocol'
            assert "'All systems are go for launch'" in result.content.storyline
            assert "'All hands, prepare for emergency protocol!'" in result.content.dialogue
            assert "'danger detected'" in result.content.dialogue

    def test_agent_with_json_arrays_containing_quotes(self):
        """Test Agent handles JSON arrays with quoted elements."""
        from unittest.mock import patch

        problematic_response = '''```json
{
    "setting": "A courtroom",
    "name": "The Verdict",
    "characters": ["Judge Wilson", "Attorney Parker", "Witness"],
    "storyline": "A trial where the witness says 'I saw everything' but the defendant claims innocence.",
    "dialogue": "The judge asked 'Are you sure about your testimony?' and the witness replied confidently.",
    "rating": {"story": 9, "acting": 8}
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You write courtroom drama scripts.",
            response_model=self.MovieScript,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(problematic_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Write a courtroom drama")
            
            assert isinstance(result.content, self.MovieScript)
            assert result.content.name == 'The Verdict'
            assert 'Judge Wilson' in result.content.characters
            assert "'I saw everything'" in result.content.storyline
            assert "'Are you sure about your testimony?'" in result.content.dialogue

    def test_agent_clean_json_passes_through_unchanged(self):
        """Test Agent with clean JSON (no quotes issues) works normally."""
        from unittest.mock import patch

        clean_response = '''```json
{
    "setting": "A peaceful meadow",
    "name": "Spring Awakening",
    "characters": ["Mary", "John"],
    "storyline": "A simple love story in the countryside without any complex dialogue.",
    "dialogue": "They walked together through the fields enjoying the sunshine.",
    "rating": {"story": 7, "acting": 6}
}
```'''

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="You write simple romance scripts.",
            response_model=self.MovieScript,
            use_json_mode=True,
        )

        with patch.object(agent.model, 'invoke') as mock_invoke:
            mock_response = self._create_mock_openai_response(clean_response)
            mock_invoke.return_value = mock_response

            result = agent.run("Write a simple romance")
            
            assert isinstance(result.content, self.MovieScript)
            assert result.content.name == 'Spring Awakening'
            assert 'Mary' in result.content.characters
            assert 'John' in result.content.characters
            assert result.content.setting == 'A peaceful meadow'
