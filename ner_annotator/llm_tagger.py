import json
import re
from pydantic import BaseModel, Field
from tqdm import tqdm
from ner_annotator.constants import (
    URDU_LETTERS_THRESHOLD,
    CHUNK_SIZE,
    MAX_CONCURRENT_REQUESTS,
)
import enum
from typing import Dict, List
from crewai import LLM
import concurrent.futures


class NERMode(enum.Enum):
    GENERAL = "general"
    MARSIYA = "marsiya"


class TaggedElement(BaseModel):
    original: str = Field(description="Original string")
    tagged: str = Field(description="String with tagged entities")
    english: str = Field(description="English translation of the original string")


class TaggedElements(BaseModel):
    """List of tagged elements"""

    tagged_elements: List[TaggedElement] = Field(description="List of tagged elements")


GENERAL_NER_SYSTEM_PROMPT = """
Perform Named Entity Recognition (NER) on the given Urdu text with strict adherence to these categories:

### Entity Categories:
1. **PERSON (شخصیت)**: Names of people, including titles if part of the name.
   - Example: `<PERSON>محمد علی جناح</PERSON>`, `<PERSON>ڈاکٹر عبدالقدیر خان</PERSON>`
   - Exclude generic titles unless attached to a name.

2. **LOCATION (مقام)**: Cities, countries, landmarks, and geographical features.
   - Example: `<LOCATION>لاہور</LOCATION>`, `<LOCATION>دریائے سندھ</LOCATION>`

3. **DATE (تاریخ)**: Specific dates, years, or named days.
   - Example: `<DATE>14 اگست 1947</DATE>`, `<DATE>یوم آزادی</DATE>`

4. **TIME (وقت)**: Specific times or periods.
   - Example: `<TIME>صبح کے 10 بجے</TIME>`, `<TIME>دوپہر</TIME>`

5. **ORGANIZATION (تنظیم)**: Companies, institutions, government bodies.
   - Example: `<ORGANIZATION>مسلم لیگ</ORGANIZATION>`, `<ORGANIZATION>لاہور ہائی کورٹ</ORGANIZATION>`

6. **DESIGNATION (لقب)**: Job titles or honorifics.
   - Example: `<DESIGNATION>وزیراعظم</DESIGNATION>`, `<DESIGNATION>ڈائریکٹر</DESIGNATION>`

7. **NUMBER (عدد)**: Important numerical values.
   - Example: `<NUMBER>50 کروڑ</NUMBER>`, `<NUMBER>تین گھنٹے</NUMBER>`

### Rules:
- Tag only clear entity mentions
- Maintain original text formatting
- Use exact XML-style tags
- For ambiguous cases, prefer more specific tags (PERSON > ORGANIZATION > LOCATION)

You also need to provide the English translation of the original string in the output.

### Output Format:
Return a list with original string, string with tagged entities and its english translation. Make sure you return the output for each line without missing any line in the text:
For example, below is the response for a text with a single line - 
{
    "tagged_elements": [
        {
            "original" : "امام حسینؑ کربلا میں 10 محرم کو شہید ہوئے۔",
            "tagged" : "<PERSON>امام حسینؑ</PERSON> <LOCATION>کربلا</LOCATION> میں <DATE>10 محرم</DATE> کو شہید ہوئے۔",
            "english" : "Imam Hussain was martyred in Karbala on 10th Muharram."
        }
    ]
}

"""

MARSIYA_NER_SYSTEM_PROMPT = """
Perform Named Entity Recognition (NER) on the given Urdu Marsiya text with strict adherence to these categories and rules:

### Entity Categories:
1. **PERSON (شخصیت)**: Names of prophets, Imams, martyrs, and historical figures. 
   - Example: `<PERSON>امام حسینؑ</PERSON>`, `<PERSON>حضرت عباسؑ</PERSON>`
   - Exclude generic terms like "شہید" unless part of a name.

2. **LOCATION (مقام)**: Sacred/historical places.
   - Example: `<LOCATION>کربلا</LOCATION>`, `<LOCATION>فرات</LOCATION>`

3. **DATE (تاریخ)**: Specific dates/Islamic months.
   - Example: `<DATE>10 محرم</DATE>`, `<DATE>یوم عاشورہ</DATE>`

4. **TIME (وقت)**: Significant time references.
   - Example: `<TIME>عصر کا وقت</TIME>`, `<TIME>طلوع فجر</TIME>`

5. **ORGANIZATION (تنظیم)**: Tribes, armies, or groups.
   - Example: `<ORGANIZATION>لشکر یزید</ORGANIZATION>`, `<ORGANIZATION>اصحاب حسینؑ</ORGANIZATION>`

6. **DESIGNATION (لقب)**: Honorific titles.
   - Example: `<DESIGNATION>سید الشہداء</DESIGNATION>`, `<DESIGNATION>قمر بنی ہاشم</DESIGNATION>`

7. **NUMBER (عدد)**: Numerals with contextual importance.
   - Example: `<NUMBER>72</NUMBER> شہداء`, `<NUMBER>تین دن</NUMBER>`

### Rules:
- Tag only explicit entities. Avoid tagging metaphors unless contextually clear.
- Use exact XML-style tags as shown.
- Preserve original Urdu text formatting (e.g., poetic verses).
- For ambiguous cases, prioritize `PERSON > DESIGNATION > ORGANIZATION`.

You also need to provide the English translation of the original string in the output.

### Output Format:
Return a list with original string, string with tagged entities and its english translation. Make sure you return the output for each line without missing any line in the text:
Example -- 

Input: 
پنڈلیاں سوجی ہیں اور طوق سے چھلتا ہے گلا 
سخت اینا میں ہے، فرزند شتہ کرب و بلا
خار تلووں میں میں مقتل سے جو پیدل ہے چلا
 دھجیاں پاؤں میں باندھے ہے وہ نازوں کا پلا
اس کی مظلومی پہ بیتاب حرم ہوتے ہیں
دیدۂ حلقۂ زنجیر لہو روتے ہیں

پیچھے بیمار کے ہے قافلہ اہل حرم
چُپ ہیں تصویر سے گویا کہ کسی میں نہیں دم
دختر فاطمہ زہرا کا عجب ہے عالم
 تھر تھری جسم میں ہے اُٹھ نہیں سکتے ہیں قدم
رو کے فرماتی ہیں کسی گوشے میں جائے زینب
ہاتھ کھل جائیں تو منھ اپنا چھپائے زینب


Output:
{
    "tagged_elements": [
        {
            "original": "پنڈلیاں سوجی ہیں اور طوق سے چھلتا ہے گلا",
            "tagged": "پنڈلیاں سوجی ہیں اور طوق سے چھلتا ہے گلا",
            "english": "The shins are swollen and the throat is wounded by the collar of chains."
        },
        {
            "original": "سخت اینا میں ہے، فرزند شتہ کرب و بلا",
            "tagged": "سخت اینا میں ہے، فرزند شتہ <LOCATION>کرب و بلا</LOCATION>",
            "english": "In severe agony is the son of the one martyred in Karbala."
        },
        {
            "original": "خار تلووں میں میں مقتل سے جو پیدل ہے چلا",
            "tagged": "خار تلووں میں میں <LOCATION>مقتل</LOCATION> سے جو پیدل ہے چلا",
            "english": "Thorns pierce his feet as he walks barefoot from the battlefield."
        },
        {
            "original": "دھجیاں پاؤں میں باندھے ہے وہ نازوں کا پلا",
            "tagged": "دھجیاں پاؤں میں باندھے ہے وہ نازوں کا پلا",
            "english": "He who was raised with affection now has rags tied to his feet."
        },
        {
            "original": "اس کی مظلومی پہ بیتاب حرم ہوتے ہیں",
            "tagged": "اس کی مظلومی پہ بیتاب <ORGANIZATION>حرم</ORGANIZATION> ہوتے ہیں",
            "english": "The women of the holy sanctuary are agitated by his oppression."
        },
        {
            "original": "دیدۂ حلقۂ زنجیر لہو روتے ہیں",
            "tagged": "دیدۂ حلقۂ زنجیر لہو روتے ہیں",
            "english": "The eyes within the circle of chains weep blood."
        },
        {
            "original": "پیچھے بیمار کے ہے قافلہ اہل حرم",
            "tagged": "پیچھے <DESIGNATION>بیمار</DESIGNATION> کے ہے قافلہ <ORGANIZATION>اہل حرم</ORGANIZATION>",
            "english": "Behind the ailing one follows the caravan of the sacred household."
        },
        {
            "original": "چُپ ہیں تصویر سے گویا کہ کسی میں نہیں دم",
            "tagged": "چُپ ہیں تصویر سے گویا کہ کسی میں نہیں دم",
            "english": "They are as silent as paintings, as if no soul remains within them."
        },
        {
            "original": "دختر فاطمہ زہرا کا عجب ہے عالم",
            "tagged": "<PERSON>دختر فاطمہ زہرا</PERSON> کا عجب ہے عالم",
            "english": "A strange state has befallen the daughter of Fatima Zahra."
        },
        {
            "original": "تھر تھری جسم میں ہے اُٹھ نہیں سکتے ہیں قدم",
            "tagged": "تھر تھری جسم میں ہے اُٹھ نہیں سکتے ہیں قدم",
            "english": "Her body trembles so much that she cannot lift her feet."
        },
        {
            "original": "رو کے فرماتی ہیں کسی گوشے میں جائے زینب",
            "tagged": "رو کے فرماتی ہیں کسی گوشے میں جائے <PERSON>زینب</PERSON>",
            "english": "She weeps and says, 'Let Zainab retreat into some corner.'"
        },
        {
            "original": "ہاتھ کھل جائیں تو منھ اپنا چھپائے زینب",
            "tagged": "ہاتھ کھل جائیں تو منھ اپنا چھپائے <PERSON>زینب</PERSON>",
            "english": "If her hands were free, Zainab would hide her face."
        }
    ]
}
"""


def is_mostly_urdu(text: str, threshold=URDU_LETTERS_THRESHOLD) -> bool:
    """
    Improved version that correctly identifies Urdu text

    Args:
        text (str): Input text to check
        threshold (float): Percentage threshold (0-1) for Urdu characters

    Returns:
        bool: True if Urdu characters meet/exceed the threshold
    """
    if not text.strip():
        return False

    if len(text.split()) < 2:
        return False

    # Define what we consider non-Urdu characters (whitespace is neutral)
    non_urdu_pattern = re.compile(
        r"[^\s\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0670-\u06D3\u06D5-\u06FF]"
    )

    # Count all non-Urdu, non-whitespace characters
    non_urdu_chars = len(non_urdu_pattern.findall(text))
    total_chars = len(text.replace(" ", ""))  # Don't count whitespace

    if total_chars == 0:
        return False

    urdu_ratio = 1 - (non_urdu_chars / total_chars)
    return urdu_ratio >= threshold


def get_ner_prompt_messages(text, mode=NERMode.MARSIYA) -> List[Dict[str, str]]:
    if mode == NERMode.GENERAL:
        system_prompt = GENERAL_NER_SYSTEM_PROMPT
    elif mode == NERMode.MARSIYA:
        system_prompt = MARSIYA_NER_SYSTEM_PROMPT
    else:
        raise ValueError("Invalid NER mode selected.")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Provide the Named entities from the below Urdu text: \n\n{text}\n\n",
        },
    ]
    return messages


def get_ner_prompt_messages_per_chunk(
    text: str, chunk_size=CHUNK_SIZE, mode=NERMode.MARSIYA
) -> List[List[Dict[str, str]]]:
    lines = [line for line in text.split("\n") if is_mostly_urdu(line)]
    chunk_messages = list()
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        chunk_messages.append(get_ner_prompt_messages(chunk, mode))

    return chunk_messages


def extract_named_entites_from_chunks(
    llm: LLM, chunks: List[List[Dict[str, str]]]
) -> List[TaggedElement]:
    """
    Extract named entities from chunks of text using the specified NER mode.

    Args:
        chunks (List[List[Dict[str, str]]]): List of chunk messages for NER processing.

    Returns:
        List[TaggedElement]: List of extracted named entities, in the same order as the input chunks.
    """
    extracted_results = [None] * len(chunks)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:
        futures = {
            executor.submit(llm.call, chunk): idx for idx, chunk in enumerate(chunks)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting NER Tags",
        ):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    extracted_results[idx] = result
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")

    # Remove any None results (if desired)
    extracted_results = [res for res in extracted_results if res is not None]

    return extracted_results


def get_ner_tags(
    text: str,
    mode=NERMode.MARSIYA,
    model_id: str = "openai/gpt-4o-mini",
    chunk_size: int = CHUNK_SIZE,
) -> TaggedElements:
    chunked_messages = get_ner_prompt_messages_per_chunk(text, chunk_size, mode)
    print("Using model:", model_id)
    print("Using chunk size:", chunk_size)
    print("Number of chunks:", len(chunked_messages))

    # with open('dataset/test_data.json') as f:
    #     return json.load(f)

    llm = LLM(model=model_id, response_format=TaggedElements)
    # responses = [llm.call(cm) for cm in tqdm(chunked_messages, desc="Processing chunks")]
    responses = extract_named_entites_from_chunks(llm, chunked_messages)
    return sum([json.loads(r)["tagged_elements"] for r in responses], [])
