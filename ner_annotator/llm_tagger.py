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
from ner_annotator.utils import get_chunks_response


class NERMode(enum.Enum):
    GENERAL = "general"
    MARSIYA = "marsiya"
    MARSIYA_ADVANCED = "marsiya_advanced"
    MARSIYA_ADVANCED_2 = "marsiya_advanced_2"


class TaggedElement(BaseModel):
    original: str = Field(description="Original string")
    tagged: str = Field(description="String with tagged entities")
    english: str = Field(description="English translation of the original string")


class TaggedElements(BaseModel):
    """List of tagged elements"""

    tagged_elements: List[TaggedElement] = Field(description="List of tagged elements")


ADVANCED_NER_SYSTEM_PROMPT_2 = """
Below is a set of practical annotation guidelines tailored for Named-Entity Recognition in Urdu Marsiya poetry. You can adapt these to your annotation interface and annotation-schema definitions.

---

## 1. Entity Types and Definitions

1. **PERSON**
   – Includes historical and religious figures (e.g., “Hazrat Imam Husayn”), poets, contemporary commentators.
   – Exclude common nouns—even if honorific-sounding—unless used as a proper name.

2. **LOCATION**
   – Sacred sites and shrines (e.g., “Imambara”, “Dargah”), geographic regions, cities, rivers.
   – Exclude metaphorical or poetic place-references (e.g., “wādī-e-shab” if used metaphorically).

3. **ORGANIZATION**
   – Religious bodies, seminaries, publishing houses (e.g., “Jamia Al-Muntazir”).
   – Do not tag generic words for assemblies or councils unless they function as named institutions.

4. **DATE**
   – Calendar dates (Gregorian or Hijri) explicitly mentioned (e.g., “10 Muharram 61 AH”).
   – Exclude vague temporal expressions (e.g., “ek roz”, “kal raat”) unless mapped to a precise date.

5. **TIME**
   – Clock times or ritual times (e.g., “sehri ka waqt”, “dopehar ke naam”).
   – Exclude general parts of day when not tied to a schedule.

6. **DESIGNATION**
   – Titles and honorifics used standalone (e.g., “Hazrat”, “Sheikh”, “Aaqa”).
   – When immediately preceding a name, nest under PERSON (see §4).

7. **RITUAL TERM** *(optional)*
   – Domain-specific terms: “Majlis”, “Noha”, “Manqabat”.
   – If in your schema, tag these; otherwise treat them as O (outside).

---

## 2. Orthography and Tokenization

* **Unicode Normalization**:
  First convert to NFC. Remove stray ZWJ/ZWNJ that disrupt token boundaries.

* **Diacritics**:
  Strip optional vowel marks, but preserve tashdīd (ّ) and ḥarakāt only when necessary for disambiguation.

* **Word Boundaries**:
  Use whitespace plus punctuation cues. For clitics (e.g., “ke”, “ka”), keep them attached to the preceding noun.

---

## 3. Span Annotation Rules

1. **Contiguous Spans**
   Tag only contiguous runs of tokens. Do not split the same name into two entity spans.

2. **Nested Entities**
   When a title or honorific appears within a PERSON span, annotate as:

   ```
   [PERSON Hazrat Imam Husayn]
   ― inside that span, DESIGNATION: Hazrat
   ```

3. **Overlapping Entities**
   If two entities overlap but belong to different classes (e.g., “Karachi Imambara Trust”), tag the larger ORGANIZATION span only. Do not create separate LOCATION “Karachi” inside it.

4. **Modifiers and Appositives**
   Appositive descriptions immediately following a name (e.g., “Imam Husayn, Shahzadah-e-Karbala”) should be included in the same span if they're part of the canonical name; otherwise, tag only the core name.

---

## 4. Handling Ambiguity

* When in doubt, consult your glossary of Marsiya-specific terms.
* If a token could be an organization or location, use context: if the verse describes a place of gathering, tag LOCATION; if it's a managing body, ORGANIZATION.

---

## 5. Special Cases

* **Poetic Epithets**:
  Epithets like “Siraj-e-Zulmat” (“Lamp of Darkness”) are considered DESIGNATION if used as a title, not PERSON.

* **Archaic Forms**:
  Map archaic inflections back to their lemma forms when possible (e.g., “Husayni” → “Husayn”), but annotate on-surface form.

* **Numeric Expressions**:
  Tag standalone numeric dates (e.g., “61”) as DATE only when clearly referring to a year; otherwise, NUMBER or O.


#### Output Format
Return a list with original string, string with tagged entities and its english translation. 
Make sure you return the output for each line without missing any line in the text:


#### Example


Example Usage:
INPUT_TEXT:
حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔
۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔
انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔


Expected JSON Output:
```
{
  "tagged_elements": [
    {
      "original": "حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔",
      "tagged": "<PERSON>حضرت زینب سلام اللہ علیہا</PERSON> نے <TIME>شامِ غریباں</TIME> میں <ORGANIZATION>خیموں</ORGANIZATION> کی تنظیم کی۔",
      "english": "Lady Zainab (peace be upon her) organized the tents during the evening of Ghariban."
    },
    {
      "original": "۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔",
      "tagged": "<DATE>۱۲ محرم ۶۱ھ</DATE> کی <TIME>صبح کے ۸ بجے</TIME> <NUMBER>سبز</NUMBER> جھنڈا لہرایا گیا۔",
      "english": "On 12 Muharram 61 AH at 8 AM, the green flag was hoisted."
    },
    {
      "original": "انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔",
      "tagged": "<ORGANIZATION>انجمنِ شعرا</ORGANIZATION> نے <LOCATION>لاہور ہائی کورٹ</LOCATION> کے سامنے <ORGANIZATION>مجلسِ سوگ</ORGANIZATION> کا اہتمام کیا۔",
      "english": "The Poets' Association organized a mourning gathering in front of the Lahore High Court."
    }
  ]
}
```

Now process the following INPUT_TEXT and output only the JSON.

"""

ADVANCED_NER_SYSTEM_PROMPT = """

You are an expert Urdu linguist and poet, specialized in Marsiya (elegiac) poetry. 
Your task is to perform Named Entity Recognition (NER) on Urdu Marsiya text. 
You must annotate the input by wrapping each entity in XML tags corresponding to the following categories:

1. **PERSON (شخصیت)**
2. **LOCATION (مقام)**
3. **DATE (تاریخ)**
4. **TIME (وقت)**
5. **ORGANIZATION (تنظیم)**
6. **DESIGNATION (لقب)**
7. **NUMBER (عدد)**

---

#### Annotation Rules

* **General**

  * Do **not** alter the original text except to insert tags.
  * Do **not** create overlapping tags (entities must be non-nested).
  * If a word or phrase fits more than one category, choose the most specific.
  * Preserve all Urdu diacritics, punctuation, and spacing.

* **PERSON (شخصیت)**

  * Include full personal names, including honorific prefixes or suffixes if they are integral (e.g. حضرت امام حسین علیہ السلام).
  * Do **not** tag generic titles alone (e.g. “ڈاکٹر” without a name → DESIGNATION).
  * Recognize compound names, Persian-Urdu hybrid names, and patronymics (e.g. ابنِ شاہنواز).

* **LOCATION (مقام)**

  * Tag cities, countries, rivers, shrines, battlefields, and other geographical places.
  * Recognize both Arabic-Persianized forms (کربلا) and Urdu colloquial names (لاہور).

* **DATE (تاریخ)**

  * Tag explicit calendar dates (e.g. “۱۴ محرم ۶۱ھ”) and named commemorations (عاشورہ, لیلة القدر).
  * Include years (in both Hijri and Gregorian) and festival names.

* **TIME (وقت)**

  * Tag clock-times (“دوپہر کے ۳ بجے”) and relative periods (“صبح”, “رات”) when they refer to a specific part of the day.

* **ORGANIZATION (تنظیم)**

  * Tag formal institutions, political parties, religious orders, majlis or “انجمنِ شعرا.”
  * Exclude generic nouns like “مجلس” unless part of a formal name.

* **DESIGNATION (لقب)**

  * Tag standalone honorifics, ranks or job titles (e.g. “وزیراعظم”, “مولوی”, “ڈی جی”).
  * If the title directly precedes or follows a PERSON, do not separate it (it belongs inside the PERSON tag).

* **NUMBER (عدد)**

  * Tag numeric expressions in words or digits (e.g. “عشرہ”, “۳۵”).
  * Include counts, durations, and proportions when salient.

---

#### Urdu-Specific Considerations

* Urdu script runs right-to-left; ensure tags do not break RTL flow.
* Recognize Urdu-Persian loanwords (e.g. شعیب, یزید) as PERSON when names.
* Handle Zero-Width Non-Joiner (ZWNJ) in compound words.

#### Marsiya-Specific Considerations

* Identify references to the Battle of Karbala and its participants (e.g. عباس, زینب) as PERSON or LOCATION.
* Tag “حرمِ امام حسین” as LOCATION even though it's a compound shrine name.
* Recognize poetic epithets (e.g. “سیدِ شہداء”) as a PERSON if attached to a name, else DESIGNATION.
* Marsiya often uses archaic or Persianized time-expressions (“پہرِ صیام”), tag them correctly as TIME.
* Recognize allusions to religious dates (محرم, صفر) in context as DATE.

---

#### Output Format
Return a list with original string, string with tagged entities and its english translation. 
Make sure you return the output for each line without missing any line in the text:

#### Example


Example Usage:
INPUT_TEXT:
حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔
۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔
انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔


Expected JSON Output:
```
{
  "tagged_elements": [
    {
      "original": "حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔",
      "tagged": "<PERSON>حضرت زینب سلام اللہ علیہا</PERSON> نے <TIME>شامِ غریباں</TIME> میں <ORGANIZATION>خیموں</ORGANIZATION> کی تنظیم کی۔",
      "english": "Lady Zainab (peace be upon her) organized the tents during the evening of Ghariban."
    },
    {
      "original": "۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔",
      "tagged": "<DATE>۱۲ محرم ۶۱ھ</DATE> کی <TIME>صبح کے ۸ بجے</TIME> <NUMBER>سبز</NUMBER> جھنڈا لہرایا گیا۔",
      "english": "On 12 Muharram 61 AH at 8 AM, the green flag was hoisted."
    },
    {
      "original": "انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔",
      "tagged": "<ORGANIZATION>انجمنِ شعرا</ORGANIZATION> نے <LOCATION>لاہور ہائی کورٹ</LOCATION> کے سامنے <ORGANIZATION>مجلسِ سوگ</ORGANIZATION> کا اہتمام کیا۔",
      "english": "The Poets' Association organized a mourning gathering in front of the Lahore High Court."
    }
  ]
}
```

Now process the following INPUT_TEXT and output only the JSON.
"""

GENERAL_NER_SYSTEM_PROMPT = """
Perform Named Entity Recognition (NER) on the given Urdu text with strict adherence to these categories:

### Entity Categories:
1. **PERSON**: Names of people, including titles if part of the name.
   - Exclude generic titles unless attached to a name.

2. **LOCATION**: Cities, countries, landmarks, and geographical features.
   
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
Return a list with original string, string with tagged entities and its english translation. 
Make sure you return the output for each line without missing any line in the text:
For example, below is the JSON response for a text - 

Example Usage:
INPUT_TEXT:
حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔
۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔
انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔


Expected JSON Output:
```
{
  "tagged_elements": [
    {
      "original": "حضرت زینب سلام اللہ علیہا نے شامِ غریباں میں خیموں کی تنظیم کی۔",
      "tagged": "<PERSON>حضرت زینب سلام اللہ علیہا</PERSON> نے <TIME>شامِ غریباں</TIME> میں <ORGANIZATION>خیموں</ORGANIZATION> کی تنظیم کی۔",
      "english": "Lady Zainab (peace be upon her) organized the tents during the evening of Ghariban."
    },
    {
      "original": "۱۲ محرم ۶۱ھ کی صبح کے ۸ بجے سبز جھنڈا لہرایا گیا۔",
      "tagged": "<DATE>۱۲ محرم ۶۱ھ</DATE> کی <TIME>صبح کے ۸ بجے</TIME> <NUMBER>سبز</NUMBER> جھنڈا لہرایا گیا۔",
      "english": "On 12 Muharram 61 AH at 8 AM, the green flag was hoisted."
    },
    {
      "original": "انجمنِ شعرا نے لاہور ہائی کورٹ کے سامنے مجلسِ سوگ کا اہتمام کیا۔",
      "tagged": "<ORGANIZATION>انجمنِ شعرا</ORGANIZATION> نے <LOCATION>لاہور ہائی کورٹ</LOCATION> کے سامنے <ORGANIZATION>مجلسِ سوگ</ORGANIZATION> کا اہتمام کیا۔",
      "english": "The Poets' Association organized a mourning gathering in front of the Lahore High Court."
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
Return a list with original string, string with tagged entities and its english translation. 
Make sure you return the output for each line without missing any line in the text:
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


Expected JSON Output:
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
    elif mode == NERMode.MARSIYA_ADVANCED:
        system_prompt = ADVANCED_NER_SYSTEM_PROMPT
    elif mode == NERMode.MARSIYA_ADVANCED_2:
        system_prompt = ADVANCED_NER_SYSTEM_PROMPT_2
    else:
        raise ValueError("Invalid NER mode selected.")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Input text: \n{text}\n",
        },
    ]
    return messages


def get_ner_prompt_messages_per_chunk(
    text: str, chunk_size=CHUNK_SIZE, mode=NERMode.MARSIYA_ADVANCED_2
) -> List[List[Dict[str, str]]]:
    lines = [line for line in text.split("\n") if is_mostly_urdu(line)]
    chunk_messages = list()
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        chunk_messages.append(get_ner_prompt_messages(chunk, mode))

    return chunk_messages


def extract_named_entites_from_text(
    llm: LLM, text: str
) -> TaggedElements:
    retries = 3
    for attempt in range(retries):
        try:
            response = llm.call(text)
            if response is not None:
                return TaggedElements.model_validate(response)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise e


def extract_named_entites_from_chunks(
    llm: LLM, chunks: List[List[Dict[str, str]]], tqdm=tqdm
) -> List[TaggedElement]:
    """
    Extract named entities from chunks of text using the specified NER mode.

    Args:
        chunks (List[List[Dict[str, str]]]): List of chunk messages for NER processing.

    Returns:
        List[TaggedElement]: List of extracted named entities, in the same order as the input chunks.
    """
    # print("LLM: ", llm.model, " API Key: ", llm.api_key)
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
    tqdm=tqdm,
) -> List[Dict]:
    chunked_messages = get_ner_prompt_messages_per_chunk(text, chunk_size, mode)
    print("Using model:", model_id)
    print("Using chunk size:", chunk_size)
    print("Number of chunks:", len(chunked_messages))

    # with open('uploads/0c6378a92283e655078bba9cec2ccaa0_marsiya_advanced.json') as f:
    #     return json.load(f)['tagged_elements']
    # import os
    # print("OPENAI API Key: ", os.environ.get("OPENAI_API_KEY", "Not Set"))
    # print("Text: ", chunked_messages[0])  # Print first 1000 characters for debugging
    # llm = LLM(model=model_id, response_format=TaggedElements, api_key=get_crew_api_key(model_id.split("/")[0]))
    print("Extracting named entities from chunks...")
    # responses = extract_named_entites_from_chunks(llm, chunked_messages, tqdm=tqdm)
    responses = get_chunks_response(
        chunked_messages=chunked_messages, 
        model=model_id.split("/")[1], 
        response_format=TaggedElements,
        tqdm=tqdm
    )
    return sum([json.loads(r)["tagged_elements"] for r in responses], [])
