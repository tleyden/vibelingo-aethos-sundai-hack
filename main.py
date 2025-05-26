from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import os
from pathlib import Path
import random
from enum import Enum
import json
from datetime import datetime

# Import OpenAI and Pydantic AI
import openai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Add Modal import at the top
import modal

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Define the Modal app
app = modal.App("vibelingo-chat")

MODAL_FILES_IGNORE = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "env/",
    "venv/",
    ".venv/",
    ".mypy_cache",
]

# Create a Modal image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .workdir("/app")
    .add_local_dir(".", remote_path="/app", ignore=MODAL_FILES_IGNORE)  # Include all local files including static directory
)

SECRETS = [
    modal.Secret.from_name("llm_api_key"),
]

# Replace the in-memory user_sessions with Modal Dict
user_sessions = modal.Dict.from_name("vibelingo-user-sessions", create_if_missing=True)

# Common system prompt for all AI interactions
SYSTEM_PROMPT = "Always speak to the user in the native language (English), not the target language (German) when correcting the users mistakes or giving instructions"

# Onboarding questions
onboarding_questions = [
    "What's your name?",
    "What topics are you passionate about or would like to explore in German? (e.g., sports, art, tech, music, travel…)",
    "How would you describe your current level of German? (Beginner / Intermediate / Advanced)",
    "Why do you want to learn German? (e.g., travel, job, study, family, personal growth…)",
    "If you could have a famous person as your German teacher, who would it be?",
]


# Request and response models
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    question: str
    completed: bool
    progress: int
    is_drill: bool = False
    word_pair: Optional[Union["VocabPair", "SentencePair"]] = None
    image_description: Optional["ImageDescription"] = None
    evaluation: Optional[Dict[str, Any]] = None
    direction: Optional[str] = None
    drill_type: Optional[str] = None


class DrillRequest(BaseModel):
    session_id: str
    message: str
    drill_type: Optional[str] = None
    feedback: Optional[str] = None
    emotion: Optional[str] = None


class DrillResponse(BaseModel):
    drill_type: str
    content: dict
    question: str
    hint: Optional[str] = None
    completed: bool = False
    feedback: Optional[str] = None


class DrillType(str, Enum):
    VOCAB = "vocab"
    SENTENCE = "sentence"
    IMAGE = "image"


class DrillDirection(str, Enum):
    DE_TO_EN = "de_to_en"
    EN_TO_DE = "en_to_de"


class VocabPair(BaseModel):
    german: str = Field(..., description="The German word or phrase")
    english: str = Field(
        ..., description="The English translation of the German word or phrase"
    )
    level: str = Field(
        ..., description="The difficulty level: beginner, intermediate, or advanced"
    )


class SentencePair(BaseModel):
    german: str = Field(..., description="The German sentence")
    english: str = Field(..., description="The English translation of the sentence")
    level: str = Field(
        ..., description="The difficulty level: beginner, intermediate, or advanced"
    )
    topic: str = Field(..., description="The topic or theme of the sentence")


class ImageDescription(BaseModel):
    image_url: str = Field(..., description="URL of the image to be described")
    prompt: str = Field(..., description="The prompt used to generate the image")
    topic: str = Field(..., description="The topic or theme of the image")
    level: str = Field(
        ..., description="The difficulty level: beginner, intermediate, or advanced"
    )
    vocabulary_de: List[str] = Field(
        ..., description="Key German vocabulary words useful for describing the image"
    )
    vocabulary_en: List[str] = Field(
        ..., description="English translations of the key vocabulary words"
    )
    hints: List[str] = Field(
        ...,
        description="Helpful hints to guide the user in describing the image in German",
    )


class AnswerEvaluationResult(BaseModel):
    correct: bool = Field(..., description="Whether the answer is correct")
    feedback: str = Field(..., description="Feedback message for the user")
    score: float = Field(..., description="Score between 0 and 1")
    explanation: str = Field(..., description="Explanation or tips to improve")


# Define functions for generating vocabulary drills and evaluating answers using Pydantic AI
def generate_vocab_drill(
    interests: List[str],
    level: str,
    previous_words: List[Dict[str, str]],
    teacher_role_model: str = "",
    conversation_history: List[Dict] = None,
) -> VocabPair:
    """Generate a vocabulary drill pair appropriate for the user's level and interests."""

    # Initialize the AI agent
    agent = Agent(model="gpt-4o", result_type=VocabPair, system_prompt=SYSTEM_PROMPT)

    interest = (
        random.choice(interests)
        if interests
        else random.choice(["sports", "art", "tech", "music", "travel"])
    )

    # Add teacher role model to the prompt if available
    teacher_instruction = ""
    if teacher_role_model:
        teacher_instruction = f"\n\nYou are a {teacher_role_model} teaching German. Match their character, teaching style, tone, and approach."

    prompt = f"""Generate a German-English vocabulary pair related to {interest} that would be appropriate for a {level} German language learner.
    
    The vocabulary should be a single word or short phrase that is commonly used and useful for conversations about {interest}.
    
    If the level is beginner, choose simple, everyday vocabulary.
    If the level is intermediate, choose moderately complex vocabulary that might be used in regular conversations.
    If the level is advanced, choose more sophisticated vocabulary that might be used in detailed discussions.{teacher_instruction}

    The level is just a rough estimate, and users are notoriously bad at estimating their own levels.  
    To further determine the user level and generate an appropriate drill, consider the conversation history:
    {conversation_history}
    
    Format your response as a Pydantic model with these fields:
    - german: The German word or phrase
    - english: The English translation
    - level: The difficulty level (beginner, intermediate, or advanced)
    """

    print(f"Vocabulary drill prompt: {prompt}")

    # Run the agent to get the response
    result = agent.run_sync(prompt)

    # Return the vocabulary pair
    return VocabPair(
        german=result.output.german,
        english=result.output.english,
        level=result.output.level,
    )


def generate_sentence_drill(
    interests: List[str],
    level: str,
    previous_sentences: List[Dict[str, str]],
    teacher_role_model: str = "",
    conversation_history: List[Dict] = None,
) -> SentencePair:
    """Generate a sentence translation drill appropriate for the user's level and interests."""
    # Create an agent with OpenAI model
    model = OpenAIModel("gpt-4o")
    agent = Agent(model, result_type=SentencePair, system_prompt=SYSTEM_PROMPT)

    print(f"Generating sentence drill for interests: {interests}, level: {level}")

    # Add teacher role model to the prompt if available
    teacher_instruction = ""
    if teacher_role_model:
        teacher_instruction = f"\n\n- Present this sentence as if you were {teacher_role_model} teaching German. Match their teaching style, tone, and approach."

    # Create the prompt for generating a sentence drill
    prompt = f"""Generate a German-English sentence pair based on the following criteria:
    - User interests: {", ".join(interests)}
    - User German level: {level}
    - The sentence should be appropriate for the user's level
    - For beginner level: simple present tense, basic vocabulary, 5-7 words
    - For intermediate level: past tense or future tense, more complex vocabulary, 8-12 words
    - For advanced level: complex sentence structure, subjunctive, specialized vocabulary, 12+ words
    - The sentence should be natural and useful in everyday conversation
    - Choose a topic related to the user's interests{teacher_instruction}

    The level is just a rough estimate, and users are notoriously bad at estimating their own levels.  
    To further determine the user level and generate an appropriate drill, consider the conversation history:
    {conversation_history}
    
    Return a structured object with these fields:
    - german: The German sentence
    - english: The English translation
    - level: The difficulty level (beginner, intermediate, or advanced)
    - topic: The topic or theme of the sentence
    """

    print(f"Sentence drill prompt: {prompt}")

    # Run the agent to get the response
    result = agent.run_sync(prompt)
    print(f"Sentence drill result: {result}")
    return result.data


def generate_image_drill(
    interests: List[str],
    level: str,
    teacher_role_model: str = "",
    conversation_history: List[Dict] = None,
) -> ImageDescription:
    """Generate an image description drill with DALL-E generated image."""
    # Create an agent with OpenAI model for generating the image description
    model = OpenAIModel("gpt-4o")
    description_agent = Agent(model, system_prompt=SYSTEM_PROMPT)

    print(
        f"Generating image description drill for interests: {interests}, level: {level}"
    )

    # First, generate an appropriate image description based on user interests and level
    interest = (
        random.choice(interests)
        if interests
        else random.choice(["nature", "city", "animals", "food", "technology"])
    )

    # Add teacher role model to the prompt if available
    teacher_instruction = ""
    if teacher_role_model:
        teacher_instruction = f"\n\nThe exercise will be presented as if {teacher_role_model} is teaching German. The image and vocabulary should match their teaching style."

    # Create a prompt for generating an image description
    description_prompt = f"""Generate a detailed description for an image that will be used in a German language learning exercise.
    The image should be related to: {interest}
    The description should be appropriate for a {level} German language learner.{teacher_instruction}

    The level is just a rough estimate, and users are notoriously bad at estimating their own levels.  
To further determine the user level and generate an appropriate drill, consider the conversation history:
{conversation_history}
    
    The description should be visually rich, clear, and suitable for image generation.
    Make sure it's appropriate for all ages and not controversial.
    Keep the description between 2-3 sentences.
    """

    # Get the image description
    print(f"Image description prompt: {description_prompt}")
    description_result = description_agent.run_sync(description_prompt)
    image_description = description_result.output
    print(f"Generated image description: {image_description}")

    # Now use DALL-E to generate the actual image using direct OpenAI API call
    # Create an OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Generate the image using OpenAI's Image API (new syntax for v1.0.0+)
    response = client.images.generate(
        model="dall-e-3",  # Use DALL-E 3 model
        prompt=image_description,
        n=1,  # Generate 1 image
        size="1024x1024",  # Standard size
    )

    # Get the image URL from the response
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

    # Now generate vocabulary and hints for the image description drill
    vocab_agent = Agent(model, system_prompt=SYSTEM_PROMPT)

    # Add teacher role model to the vocabulary prompt if available
    teacher_vocab_instruction = ""
    if teacher_role_model:
        teacher_vocab_instruction = f"\n\nPresent the vocabulary and hints as if {teacher_role_model} is teaching German. Match their teaching style, tone, and approach."

    vocab_prompt = f"""Based on this image description: "{image_description}"
    
    Generate vocabulary and hints for a German language learner at {level} level who will be describing this image.{teacher_vocab_instruction}
    
    Return a JSON object with these fields:
    - vocabulary_de: A list of 5-7 German words or phrases useful for describing this image
    - vocabulary_en: The English translations of these words in the same order
    - hints: 3-4 helpful hints in English for describing the image in German
    - topic: A short phrase (2-3 words) describing the theme of the image
    """

    print(f"Vocabulary drill prompt: {vocab_prompt}")

    vocab_result = vocab_agent.run_sync(vocab_prompt)

    # Try to parse the JSON from the response
    # Check if we have content or output attribute
    if hasattr(vocab_result, "content"):
        vocab_text = vocab_result.content
    else:
        vocab_text = vocab_result.output

    # Clean up the text to ensure it's valid JSON
    vocab_text = vocab_text.strip()
    if vocab_text.startswith("```json"):
        vocab_text = vocab_text.split("```json", 1)[1]
    if vocab_text.endswith("```"):
        vocab_text = vocab_text.rsplit("```", 1)[0]
    vocab_text = vocab_text.strip()

    # Parse the JSON
    vocab_data = json.loads(vocab_text)

    # Create the ImageDescription object
    return ImageDescription(
        image_url=image_url,
        prompt=image_description,
        topic=vocab_data.get("topic", interest),
        level=level,
        vocabulary_de=vocab_data.get("vocabulary_de", []),
        vocabulary_en=vocab_data.get("vocabulary_en", []),
        hints=vocab_data.get("hints", []),
    )


def evaluate_answer(
    user_answer: str,
    correct_answer: str = None,
    direction: DrillDirection = None,
    user_level: str = None,
    drill_type: DrillType = DrillType.VOCAB,
    image_description: ImageDescription = None,
    teacher_role_model: str = "",
    conversation_history: List[Dict] = None,
) -> Dict[str, Any]:
    """Evaluate the user's answer and provide feedback."""
    # Create an agent with OpenAI model
    model = OpenAIModel("gpt-4o")
    agent = Agent(
        model, result_type=AnswerEvaluationResult, system_prompt=SYSTEM_PROMPT
    )

    # Add teacher role model to the prompt if available
    teacher_instruction = ""
    if teacher_role_model:
        teacher_instruction = f"\n\nYou are {teacher_role_model} teaching German. Provide feedback in their character, teaching style, tone, and approach."

    if drill_type == DrillType.IMAGE and image_description:
        # For image descriptions, we evaluate differently
        prompt = f"""Evaluate this German language image description:
        - User's description: "{user_answer}"
        - Image prompt: "{image_description.prompt}"
        - User's German level: {user_level}
        - Topic: {image_description.topic}
        - Key vocabulary that could be used, but do not have to be used: {", ".join(image_description.vocabulary_de)}{teacher_instruction}
        
        For image descriptions, evaluate based on:
        1. Whether the user attempted to describe the main elements that might be in the image
        2. Appropriate use of German vocabulary and grammar for their level
        3. Use of some of the suggested vocabulary words
        4. Overall communicative effectiveness, not perfect grammar
        5. It does not have to be exhaustive and cover all aspects of the image, just the central aspect
        6. Be very generous in your grading, and do not require usage of all vocabulary words
        
        Return a JSON object with these fields:
        - correct: Whether the answer is generally correct (true/false)
        - feedback: A short, encouraging feedback message
        - score: A score between 0 and 1
        - explanation: Brief explanation or tips to improve
        """
    else:
        # For vocabulary and sentence drills
        if direction == DrillDirection.DE_TO_EN:
            # German to English
            prompt = f"""Evaluate this German to English translation:
            - German: "{correct_answer}"
            - User's translation: "{user_answer}"
            - User's German level: {user_level}{teacher_instruction}
            
            Return a JSON object with these fields:
            - correct: Whether the translation is correct (true/false), allowing for minor typos or slight variations
            - feedback: A short, encouraging feedback message
            - score: A score between 0 and 1
            - explanation: If the answer is wrong, provide a brief explanation
            """
        else:
            # English to German
            prompt = f"""Evaluate this English to German translation:
            - English: "{correct_answer}"
            - User's translation: "{user_answer}"
            - User's German level: {user_level}{teacher_instruction}
            
            For {user_level} level, be more lenient with:
            - Articles (der/die/das)
            - Minor spelling mistakes
            - Word order
            
            Return a JSON object with these fields:
            - correct: Whether the translation is correct (true/false), allowing for minor mistakes appropriate for level
            - feedback: A short, encouraging feedback message
            - score: A score between 0 and 1
            - explanation: If the answer is wrong, provide a brief explanation
            """

    # Run the agent to get the structured response
    print(f"Evaluation prompt: {prompt}")
    result = agent.run_sync(prompt)
    print(f"Evaluation result: {result}")

    # Convert the Pydantic model to a dictionary
    return result.data.dict()


@app.function(image=image, secrets=SECRETS)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI()

    # Configure CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Mount static files directory
    static_dir = Path("/app/static")
    web_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Add a root endpoint to serve the index.html
    @web_app.get("/")
    async def get_index():
        return RedirectResponse(url="/static/index.html")

    @web_app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        session_id = request.session_id
        message = request.message

        # Initialize session if it doesn't exist
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                "current_question": 0,
                "onboarding_answers": [],
                "drill_active": False,
                "current_drill": None,
                "drill_evaluations": [],  # Store evaluations for analysis
                "drill_history": [],  # Store completed drills
                "conversation_history": [],  # Store all interactions
                "performance_metrics": {  # Track user performance
                    "vocab": {
                        "total_attempts": 0,
                        "correct_answers": 0,
                        "average_score": 0.0,
                        "common_mistakes": [],
                    },
                    "sentence": {
                        "total_attempts": 0,
                        "correct_answers": 0,
                        "average_score": 0.0,
                        "common_mistakes": [],
                    },
                    "image": {
                        "total_attempts": 0,
                        "average_score": 0.0,
                        "vocabulary_usage": [],
                    },
                },
                "user_level": "beginner",  # Default level
                "user_interests": [],  # Will be populated from onboarding
                "teacher_role_model": "",  # Will be populated from onboarding
            }

        session = user_sessions[session_id]

        # Check if we're in drill mode or onboarding mode
        if session.get("drill_active", False):
            return handle_drill(session_id, message)
        else:
            # We're in onboarding mode
            current_question = session["current_question"]

            # Store the answer if not the first question (which is just starting)
            if current_question > 0 and message:
                session["onboarding_answers"].append(message)

                # Process the answer based on the question
                if current_question == 1:
                    session["teacher_role_model"] = message
                elif current_question == 2:  # Interests question
                    # Extract interests from the message
                    interests = [
                        interest.strip().lower() for interest in message.split(",")
                    ]
                    session["user_interests"] = interests
                elif current_question == 3:  # Level question
                    # Extract level from the message
                    if "beginner" in message.lower():
                        session["user_level"] = "beginner"
                    elif "intermediate" in message.lower():
                        session["user_level"] = "intermediate"
                    elif "advanced" in message.lower():
                        session["user_level"] = "advanced"

                # Move to the next question
                current_question += 1
                session["current_question"] = current_question
                
                # Update session in Modal Dict
                user_sessions[session_id] = session
            else:
                # For the first interaction, just increment to the first question
                session["current_question"] = 1
                # Update session in Modal Dict
                user_sessions[session_id] = session

            print(f"User session: {session}")

            # Check if all questions have been answered
            if current_question >= len(onboarding_questions):
                # Transition to drill mode
                session["drill_active"] = True
                # Update session in Modal Dict
                user_sessions[session_id] = session

                # Generate the first drill
                return start_drill(session_id)

            # Return the next onboarding question
            return ChatResponse(
                question=onboarding_questions[current_question - 1],
                completed=False,
                progress=current_question,
            )

    @web_app.get("/session/{session_id}")
    def get_session(session_id: str):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        return user_sessions[session_id]

    return web_app


def handle_drill(session_id: str, message: str) -> ChatResponse:
    """Handle a user's response to a vocabulary, sentence, or image drill."""
    session = user_sessions[session_id]
    current_drill = session.get("current_drill")

    if not current_drill:
        # No active drill, start a new one
        return start_drill(session_id)

    # Get drill information
    drill_type = current_drill["drill_type"]
    user_answer = message.strip()
    user_level = session["user_level"]

    # Get conversation history
    conversation_history = session.get("conversation_history", [])

    # Handle different drill types
    # Get the teacher role model from the session
    teacher_role_model = session.get("teacher_role_model", "")

    if drill_type == DrillType.IMAGE:
        # For image description drills
        image_description = current_drill["content"]

        # Evaluate the user's image description
        evaluation = evaluate_answer(
            user_answer=user_answer,
            user_level=user_level,
            drill_type=DrillType.IMAGE,
            image_description=image_description,
            teacher_role_model=teacher_role_model,
            conversation_history=conversation_history,
        )

        # Store the image description drill record
        content_dict = {
            "image_url": image_description.image_url,
            "prompt": image_description.prompt,
            "topic": image_description.topic,
            "level": image_description.level,
            "vocabulary_de": image_description.vocabulary_de,
            "vocabulary_en": image_description.vocabulary_en,
            "hints": image_description.hints,
        }

        drill_record = {
            "drill_type": drill_type,
            "content": content_dict,
            "user_answer": user_answer,
            "evaluation": evaluation,
            "timestamp": str(datetime.now()),
        }
    else:
        # For vocabulary and sentence drills
        direction = current_drill["direction"]
        content = current_drill["content"]

        # Get the correct answer based on direction
        if direction == DrillDirection.DE_TO_EN:
            correct_answer = content.english
        else:  # EN_TO_DE
            correct_answer = content.german

        # Evaluate the answer
        evaluation = evaluate_answer(
            user_answer=user_answer,
            correct_answer=correct_answer,
            direction=direction,
            user_level=user_level,
            drill_type=drill_type,
            teacher_role_model=teacher_role_model,
            conversation_history=conversation_history,
        )

        # Store the drill record
        if drill_type == DrillType.VOCAB:
            content_dict = {
                "german": content.german,
                "english": content.english,
                "level": content.level,
            }
        else:  # SENTENCE
            content_dict = {
                "german": content.german,
                "english": content.english,
                "level": content.level,
                "topic": content.topic,
            }

        drill_record = {
            "drill_type": drill_type,
            "content": content_dict,
            "direction": direction,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "evaluation": evaluation,
            "timestamp": str(datetime.now()),
        }

    conversation_entry = {
        "role": "user",
        "content": user_answer,
        "drill_type": drill_type,
        "evaluation": evaluation,
    }
    conversation_history.append(conversation_entry)
    session["conversation_history"] = conversation_history

    # Store the evaluation and drill history
    session["drill_evaluations"].append(evaluation)
    session["drill_history"].append(drill_record)

    # Update session in Modal Dict
    user_sessions[session_id] = session

    # Generate the next drill
    return start_drill(
        session_id, show_previous_evaluation=True, previous_evaluation=evaluation
    )


def start_drill(
    session_id: str,
    show_previous_evaluation: bool = False,
    previous_evaluation: Dict = None,
) -> ChatResponse:
    """Start a new drill (vocabulary, sentence, or image description)."""
    session = user_sessions[session_id]
    print(f"Starting drill for session {session_id}")

    # Get user interests, level, teacher role model, and conversation history from session
    interests = session.get("user_interests", [])
    level = session.get("user_level", "beginner")
    teacher_role_model = session.get("teacher_role_model", "")
    conversation_history = session.get("conversation_history", [])

    # Decide which type of drill to use (rotate between vocab, sentence, and image)
    drill_history = session.get("drill_history", [])

    if not drill_history:
        # First drill is always vocabulary
        drill_type = DrillType.VOCAB
    else:
        last_drill_type = drill_history[-1].get("drill_type")
        if last_drill_type == DrillType.VOCAB:
            drill_type = DrillType.SENTENCE
        elif last_drill_type == DrillType.SENTENCE:
            drill_type = DrillType.IMAGE
        else:  # last was IMAGE
            drill_type = DrillType.VOCAB

    # Generate the appropriate drill content
    if drill_type == DrillType.VOCAB:
        content = generate_vocab_drill(
            interests=interests,
            level=level,
            previous_words=[],
            teacher_role_model=teacher_role_model,
            conversation_history=conversation_history,
        )
        print(f"Generated vocabulary drill: {content}")

        # Randomly choose direction: German to English or English to German
        direction = random.choice([DrillDirection.DE_TO_EN, DrillDirection.EN_TO_DE])

        # Store the current drill
        session["current_drill"] = {
            "drill_type": drill_type,
            "content": content,
            "direction": direction,
        }

        # Update session in Modal Dict
        user_sessions[session_id] = session

        # Create the question
        if direction == DrillDirection.DE_TO_EN:
            question = f"Translate to English: {content.german}"
        else:  # EN_TO_DE
            question = f"Translate to German: {content.english}"

        # Add previous evaluation feedback if available
        if show_previous_evaluation and previous_evaluation:
            question = f"{previous_evaluation['feedback']}\n\n{question}"

        return ChatResponse(
            question=question,
            completed=False,
            progress=len(onboarding_questions),
            is_drill=True,
            word_pair=content,
            direction=direction,
            drill_type=drill_type,
            evaluation=previous_evaluation if show_previous_evaluation else None,
        )

    elif drill_type == DrillType.SENTENCE:
        content = generate_sentence_drill(
            interests=interests,
            level=level,
            previous_sentences=[],
            teacher_role_model=teacher_role_model,
            conversation_history=conversation_history,
        )
        print(f"Generated sentence drill: {content}")

        # Randomly choose direction: German to English or English to German
        direction = random.choice([DrillDirection.DE_TO_EN, DrillDirection.EN_TO_DE])

        # Store the current drill
        session["current_drill"] = {
            "drill_type": drill_type,
            "content": content,
            "direction": direction,
        }

        # Update session in Modal Dict
        user_sessions[session_id] = session

        # Create the question
        if direction == DrillDirection.DE_TO_EN:
            question = f"Translate to English:\n\n{content.german}"
        else:  # EN_TO_DE
            question = f"Translate to German:\n\n{content.english}"

        # Add previous evaluation feedback if available
        if show_previous_evaluation and previous_evaluation:
            question = f"{previous_evaluation['feedback']}\n\n{question}"

        return ChatResponse(
            question=question,
            completed=False,
            progress=len(onboarding_questions),
            is_drill=True,
            word_pair=content,
            direction=direction,
            drill_type=drill_type,
            evaluation=previous_evaluation if show_previous_evaluation else None,
        )

    else:  # DrillType.IMAGE
        content = generate_image_drill(
            interests=interests,
            level=level,
            teacher_role_model=teacher_role_model,
            conversation_history=conversation_history,
        )
        print(f"Generated image drill: {content}")

        # Store the current drill
        session["current_drill"] = {
            "drill_type": drill_type,
            "content": content,
            # No direction for image drills
        }

        # Update session in Modal Dict
        user_sessions[session_id] = session

        # Create the question
        vocabulary_list = "\n".join(
            [
                f"• {de} - {en}"
                for de, en in zip(content.vocabulary_de, content.vocabulary_en)
            ]
        )
        hints_list = "\n".join([f"• {hint}" for hint in content.hints])

        # Format the question with the image at the top - don't include the image prompt
        question = f"Describe this image in German:\n\n![Image]({content.image_url})\n\n**Helpful vocabulary:**\n{vocabulary_list}\n\n**Hints:**\n{hints_list}"

        # Add previous evaluation feedback if available
        if show_previous_evaluation and previous_evaluation:
            question = f"{previous_evaluation['feedback']}\n\n{question}"

        return ChatResponse(
            question=question,
            completed=False,
            progress=len(onboarding_questions),
            is_drill=True,
            image_description=content,  # Use image_description field instead of word_pair
            drill_type=drill_type,
            evaluation=previous_evaluation if show_previous_evaluation else None,
        )


