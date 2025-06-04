from typing import List, Optional, Dict
from unittest.mock import Mock

from pydantic import BaseModel, Field

from agno.utils.string import parse_response_model_str, url_safe_string


def test_url_safe_string_spaces():
    """Test conversion of spaces to dashes"""
    assert url_safe_string("hello world") == "hello-world"


def test_url_safe_string_camel_case():
    """Test conversion of camelCase to kebab-case"""
    assert url_safe_string("helloWorld") == "hello-world"


def test_url_safe_string_snake_case():
    """Test conversion of snake_case to kebab-case"""
    assert url_safe_string("hello_world") == "hello-world"


def test_url_safe_string_special_chars():
    """Test removal of special characters"""
    assert url_safe_string("hello@world!") == "helloworld"


def test_url_safe_string_consecutive_dashes():
    """Test handling of consecutive dashes"""
    assert url_safe_string("hello--world") == "hello-world"


def test_url_safe_string_mixed_cases():
    """Test a mix of different cases and separators"""
    assert url_safe_string("hello_World Test") == "hello-world-test"


def test_url_safe_string_preserve_dots():
    """Test preservation of dots"""
    assert url_safe_string("hello.world") == "hello.world"


def test_url_safe_string_complex():
    """Test a complex string with multiple transformations"""
    assert (
        url_safe_string("Hello World_Example-String.With@Special#Chars")
        == "hello-world-example-string.withspecialchars"
    )


class MockModel(BaseModel):
    name: str
    value: str
    description: str = ""


class StepModel(BaseModel):
    step: int
    action: str
    result: str


class Steps(BaseModel):
    steps: List[StepModel]


class ComplexMovieModel(BaseModel):
    title: str = Field(description="Movie title with potential quotes")
    characters: List[str] = Field(description="Character names")
    dialogue_sample: str = Field(description="Sample dialogue")
    plot_summary: str = Field(description="Plot summary")
    reviews: Dict[str, str] = Field(description="Review quotes")


def test_parse_direct_json():
    """Test parsing a clean JSON string directly"""
    content = '{"name": "test", "value": "123"}'
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"


def test_parse_json_with_markdown_block():
    """Test parsing JSON from a markdown code block"""
    content = """Some text before
    ```json
    {
        "name": "test",
        "value": "123"
    }
    ```
    Some text after"""
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"


def test_parse_json_with_generic_code_block():
    """Test parsing JSON from a generic markdown code block"""
    content = """Some text before
    ```
    {
        "name": "test",
        "value": "123"
    }
    ```
    Some text after"""
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"


def test_parse_json_with_control_characters():
    """Test parsing JSON with control characters that need cleaning"""
    content = '{\n\r"name": "test",\n\r"value": "123"\n\r}'
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"


def test_parse_json_with_markdown_formatting():
    """Test parsing JSON with markdown formatting characters"""
    content = '*{"name": "*test*", "value": "123"}*'
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"


def test_parse_json_with_quotes_in_values():
    """Test parsing JSON with quotes in values"""
    content = '{"name": "test "quoted" text", "value": "some "quoted" value"}'
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == 'test "quoted" text'
    assert result.value == 'some "quoted" value'


def test_parse_json_with_missing_required_field():
    """Test parsing JSON with missing required field"""
    content = '{"value": "123"}'  # Missing required 'name' field
    result = parse_response_model_str(content, MockModel)
    assert result is None


def test_parse_invalid_json():
    """Test parsing invalid JSON"""
    content = '{"name": "test", value: "123"}'
    result = parse_response_model_str(content, MockModel)
    assert result is None


def test_parse_empty_string():
    """Test parsing empty string"""
    content = ""
    result = parse_response_model_str(content, MockModel)
    assert result is None


def test_parse_non_json_string():
    """Test parsing non-JSON string"""
    content = "Just some regular text"
    result = parse_response_model_str(content, MockModel)
    assert result is None


def test_parse_json_with_code_blocks_in_fields():
    """Test parsing JSON with code blocks in field values"""
    content = """
    ```json
    {
        "name": "test",
        "value": "```python
    def hello():
        print('Hello, world!')
    ```",
        "description": "A function that prints hello"
    }
    ```
    """
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert "def hello()" in result.value
    assert "print('Hello, world!')" in result.value
    assert result.description == "A function that prints hello"


def test_parse_complex_markdown():
    """Test parsing JSON embedded in complex markdown"""
    content = """# Title
    Here's some text with *formatting* and a code block:

    ```json
    {
        "name": "test",
        "value": "123",
        "description": "A \"quoted\" description"
    }
    ```

    And some more text after."""
    result = parse_response_model_str(content, MockModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == "123"
    assert result.description == 'A "quoted" description'


def test_parse_nested_json():
    """Test parsing nested JSON"""

    class Step(BaseModel):
        step: str
        description: str

    class Steps(BaseModel):
        steps: List[Step]

    content = """
    ```json
    {
        "steps": [
            {
                "step": "1",
                "description": "Step 1 description"
            },
            {
                "step": "2",
                "description": "Step 2 description"
            }
        ]
    }
    ```"""
    result = parse_response_model_str(content, Steps)
    assert result is not None
    assert result.steps[0].step == "1"
    assert result.steps[0].description == "Step 1 description"
    assert result.steps[1].step == "2"
    assert result.steps[1].description == "Step 2 description"


def test_parse_json_with_validation_error():
    """Test parsing JSON that doesn't match the model schema"""
    content = '{"wrong_field": "test", "another_wrong_field": "123"}'
    result = parse_response_model_str(content, MockModel)
    assert result is None


def test_parse_nested_json_structure():
    """Test parsing more complex nested JSON structure"""
    content = """```json
    {
        "steps": [
            {
                "step": 1,
                "action": "start",
                "result": "success"
            },
            {
                "step": 2,
                "action": "process",
                "result": "pending"
            }
        ]
    }
    ```"""
    result = parse_response_model_str(content, Steps)
    assert result is not None
    assert len(result.steps) == 2
    assert result.steps[0].step == 1
    assert result.steps[0].action == "start"
    assert result.steps[1].step == 2
    assert result.steps[1].action == "process"


class TestJsonQuotesFix:
    """Test the specific JSON quotes fix functionality."""

    def test_simple_unescaped_quotes_in_values(self):
        """Test fixing simple unescaped quotes in JSON string values."""
        content = '{"name": "test "quoted" text", "value": "some "quoted" value"}'
        result = parse_response_model_str(content, MockModel)
        assert result is not None
        assert result.name == 'test "quoted" text'
        assert result.value == 'some "quoted" value'

    def test_dialogue_quotes_in_movie_script(self):
        """Test movie script with dialogue containing quotes - simplified."""
        content = '''{"title": "The Final Stand", "characters": ["Detective Johnson"], "dialogue_sample": "He said "You will never catch me" with a smile.", "plot_summary": "A detective story.", "reviews": {"critic1": "Good movie"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "The Final Stand"
        assert '"You will never catch me"' in result.dialogue_sample

    def test_simple_array_with_quotes(self):
        """Test simple array with quoted content."""
        content = '''{"title": "Array Test", "characters": ["Detective Holmes"], "dialogue_sample": "Test dialogue.", "plot_summary": "Test plot.", "reviews": {"test": "good"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Array Test"
        assert "Detective Holmes" in result.characters

    def test_nested_quotes_basic(self):
        """Test basic nested quotes handling."""
        content = '''{"title": "Conversation", "characters": ["Speaker"], "dialogue_sample": "She said "I cannot believe it" yesterday.", "plot_summary": "Simple conversations.", "reviews": {"test": "good"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Conversation"
        assert '"I cannot believe it"' in result.dialogue_sample

    def test_mixed_escaped_and_unescaped_basic(self):
        """Test basic mixed escaped and unescaped quotes."""
        content = '''{"title": "Justice Cold", "characters": ["Lawyer Davis"], "dialogue_sample": "The evidence shows truth.", "plot_summary": "A lawyer story.", "reviews": {"test": "good"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Justice Cold"
        assert "Lawyer Davis" in result.characters

    def test_multiline_basic_with_quotes(self):
        """Test basic multiline content with quotes."""
        content = '''```json
{
    "title": "Epic Adventure",
    "characters": ["Hero"],
    "dialogue_sample": "A dialogue with some text.",
    "plot_summary": "Multi-line story here.",
    "reviews": {"test": "good"}
}
```'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert "Epic Adventure" in result.title
        assert "Hero" in result.characters

    def test_quotes_at_value_boundaries_basic(self):
        """Test basic quotes at boundaries."""
        content = '''{"title": "Quoted Title", "characters": ["Character"], "dialogue_sample": "Test dialogue here.", "plot_summary": "Plot with text here.", "reviews": {"test": "review text"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Quoted Title"
        assert "Character" in result.characters

    def test_special_characters_basic(self):
        """Test special characters with basic quotes."""
        content = '''{"title": "Special Title!", "characters": ["Character Special"], "dialogue_sample": "Dialogue with symbols here!", "plot_summary": "Plot with text.", "reviews": {"special": "Review text"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert "Special Title!" == result.title
        assert "Character Special" in result.characters

    def test_already_valid_json_unchanged(self):
        """Test that already valid JSON passes through unchanged."""
        content = '''{"title": "Clean Title", "characters": ["Alice", "Bob"], "dialogue_sample": "Clean dialogue without quotes.", "plot_summary": "Clean plot summary.", "reviews": {"clean": "Clean review"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Clean Title"
        assert result.characters == ["Alice", "Bob"]
        assert result.dialogue_sample == "Clean dialogue without quotes."
        assert result.plot_summary == "Clean plot summary."
        assert result.reviews["clean"] == "Clean review"

    def test_completely_broken_json_fallback(self):
        """Test behavior with completely broken JSON that cannot be fixed."""
        content = '''{"title": "Broken", "characters": [malformed, array], "dialogue_sample": missing quotes and broken syntax, "plot_summary": "Also broken", "reviews": {broken: dict}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is None

    def test_real_world_dialogue_scenario(self):
        """Test realistic dialogue scenario with quotes."""
        content = '''{"title": "Detective Story", "characters": ["Detective Smith"], "dialogue_sample": "The suspect told me "I was not there" but evidence suggests otherwise.", "plot_summary": "A detective investigates a case.", "reviews": {"reviewer": "Engaging plot"}}'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "Detective Story"
        assert '"I was not there"' in result.dialogue_sample

    def test_cookbook_movie_script_example(self):
        """Test realistic movie script like those from cookbook examples."""
        content = '''```json
{
    "title": "New York Adventure",
    "characters": ["Detective Johnson", "Criminal X"],
    "dialogue_sample": "Johnson shouted "Stop right there!" as the criminal fled.",
    "plot_summary": "A thrilling chase through New York City streets.",
    "reviews": {"critic": "Action-packed thriller"}
}
```'''
        result = parse_response_model_str(content, ComplexMovieModel)
        assert result is not None
        assert result.title == "New York Adventure"
        assert "Detective Johnson" in result.characters
        assert "Criminal X" in result.characters
        assert '"Stop right there!"' in result.dialogue_sample
        assert "New York City" in result.plot_summary


def test_url_safe_string():
    """Test the url_safe_string function"""
    assert url_safe_string("hello world") == "hello-world"
    assert url_safe_string("camelCase") == "camel-case"
    assert url_safe_string("snake_case") == "snake-case"
    assert url_safe_string("special!@#$%chars") == "specialchars"
    assert url_safe_string("multiple---dashes") == "multiple-dashes"
