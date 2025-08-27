from openai import OpenAI
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


audio_path = "new.m4a"


with open(audio_path, "rb") as f:
    t = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="text",
        temperature=0
    )

print("Raw Whisper Output:", t)
print("=" * 50)

def trans(text,lang):
    """Give a translation of the word or sentence to the specified language asked"""
    return f"Translate the following text to {lang}:\n\n{text}"

help=Agent(
    role="You are a translator and you translate english into any language asked",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")),
    instructions=[
        "You are a helpful assistant that translates English text into any language specified by the user.",
        "When given a piece of text and a target language, provide an accurate translation in that language.",
        "If the target language is not specified, default to translating into Spanish.",
        "Ensure the translation maintains the original meaning and context of the text.",
        "Respond only with the translated text, without any additional commentary or formatting."   
    ],
    show_tool_calls=False,
    markdown=False,
    stream=False
)


voice = Agent(
    name="Language Enhancement Agent",
    role="Text Refiner and Grammar Corrector",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")),
    instructions=[
        "You are an expert language enhancement specialist. Your task is to improve the given text in the following ways:",
        "",
        "1. GRAMMAR & TENSE CORRECTION:",
        "- Fix all grammatical errors including verb tenses, subject-verb agreement, and sentence structure",
        "- Ensure proper punctuation and capitalization",
        "",
        "2. VOCABULARY ENHANCEMENT:",
        "- Replace simple words with more sophisticated alternatives (e.g., 'good' → 'excellent', 'bad' → 'inadequate')", 
        "- Use more precise and descriptive language",
        "- Replace casual expressions with formal equivalents (e.g., 'a lot of' → 'numerous', 'really' → 'considerably')",
        "",
        "3. SENTENCE STRUCTURE IMPROVEMENT:",
        "- Combine choppy sentences into more fluid, complex sentences where appropriate",
        "- Use varied sentence structures for better flow",
        "- Add transitional phrases for better coherence",
        "",
        "4. FORMAL REGISTER:",
        "- Convert casual speech patterns to formal written English",
        "- Remove filler words and redundancies",
        "- Maintain the original meaning while elevating the language level",
        "",
        "IMPORTANT: Return ONLY the enhanced text without any explanations, comments, or additional formatting.",
        "The output should be ready to use as polished, professional text."
    ],
    show_tool_calls=False,
    markdown=False,
    stream=False
)


response = voice.run(t)


if hasattr(response, 'content'):
    enhanced_text = response.content
elif hasattr(response, 'text'):
    enhanced_text = response.text  
elif hasattr(response, 'response'):
    enhanced_text = response.response
else:
    enhanced_text = str(response)

print("Enhanced Output:")
print(enhanced_text)


print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"ORIGINAL: {t}")
print("-" * 40)
print(f"ENHANCED: {enhanced_text}")