import time
from crewai import LLM
import concurrent.futures

from ner_annotator.utils import format_llm_response
from settings import MAX_CONCURRENT_REQUESTS
from typing import List, Dict
from tqdm.auto import tqdm
from pydantic import BaseModel, Field


class Entity(BaseModel):
    entity: str = Field(description="Entity text")
    tag: str = Field(description="Original Entity tag")
    score: float = Field(default=1.0, description="Confidence score of the entity")
    justification: str = Field(description="Justification for the entity tagging")
    correct: bool = Field(description="Is the entity correctly tagged?")
    alternative: str = Field(description="Alternative entity text if correctness is false")

class LLMJudgement(BaseModel):
    predictions: List[Entity] = Field(description="List of predicted entities")
    


CONTEXT_LENGTH = 3
SENTENCES_CHUNK = 10


NER_JUDGE_SYSTEM_PROMPT_REFINED = """
You are an expert Urdu linguist and Marsiya poet, and your task is to **evaluate** a set of predicted Named Entity tags on Urdu Marsiya text. 
For each extracted entity you must decide if its **Predicted NER Tag** is correct, and if not supply the correct one.

**Output Schema**
Return **only** this JSON structure:

```json
{
  "predictions": [
    {
      "entity": "…",            
      "tag": "…",               
      "correct": true|false,    
      "alternative": "…"        
    },
    …
  ]
}
```

* **entity**: the exact Urdu string tagged
* **tag**: the model's predicted tag (one of PERSON, LOCATION, DATE, TIME, ORGANIZATION, DESIGNATION, NUMBER)
* **score**: confidence score (default 1.0)
* **justification**: a brief explanation of why the tag is correct or incorrect
* **correct**: `true` if the prediction is valid, else `false`
* **alternative**: if `correct=true`, repeat the same tag; if `false`, give the one correct tag

---

### Evaluation Rules

1. **General**

   * Do **not** invent entities or tags; only judge the provided predictions.
   * No overlapping or nested entities.
   * If an entity spans multiple words (e.g. titles + names), treat as one.

2. **Urdu-Specific**

   * Honorific prefixes/suffixes (حضرت, علیہ السلام) attach to PERSON.
   * Loanwords (شعیب, یزید) are PERSON if names.
   * ZWNJ and diacritics may split tokens—treat them correctly.

3. **Marsiya-Specific**

   * Battle-of-Karbala participants (حسین، عباس، زینب) are always PERSON.
   * Shrines and holy precincts (حرمِ امام حسین) are LOCATION.
   * Poetic epithets (سیدِ شہداء) → PERSON if attached to a name, else DESIGNATION.
   * Marsiya dates (محرم, صفر, لیلة القدر) → DATE.
   * Archaic time expressions (“شامِ غریباں”, “پہرِ صیام”) → TIME.

4. **Category Quick-Reference**

   * **PERSON**: full names, epithets, honorifics + name
   * **LOCATION**: cities, rivers, shrines, battlefields
   * **DATE**: calendar dates, years (هجری/میلادی), religious commemoration days
   * **TIME**: clock-times, parts of day when specific
   * **ORGANIZATION**: formal bodies (majlis, انجمن, political/religious orders)
   * **DESIGNATION**: standalone titles/ranks not prefixed by a name
   * **NUMBER**: numerals, spelled-out counts/durations

---

### Example Evaluation

```text
Entity: علم دار حسینی
Predicted NER Tag: PERSON
```

* This is a **poetic epithet** (“Standard-Bearer of Husayn”), not a personal name.
* **Output**:

  ```json
  {
    "entity": "علم دار حسینی",
    "tag": "PERSON",
    "correct": false,
    "alternative": "DESIGNATION"
  }
  ```

Now evaluate the full list below according to these rules and return the JSON.
"""


NER_JUDGE_SYSTEM_PROMPT = """
You are an expert in Urdu Named Entity Recognition (NER). Your task is to evaluate named entities for accuracy.

Below are all the sentences and their corresponding tagged entities. Your task is to evaluate the tagged entities and provide feedback on their accuracy.
For each sentence, you need to provide the following:
1. entity - Entity name
2. correct - Whether the entity is tagged correctly or not. (True/False)
3. alternative - If correct is False, provide the correct tag for the entity. If correct is True, provide the same tag as the original.

The values of alternative MUST ONLY BE one of the following:

The possible tags are:
1. PERSON
2. LOCATION
3. DATE
4. TIME
5. ORGANIZATION
6. DESIGNATION
7. NUMBER

### Example:

---BEGINNING OF SENTENCES---
Original Urdu Text, with tags:
مرثیہ <PERSON>میر انیس</PERSON>
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
1. Entity: میر انیس
Predicted NER Tag: PERSON

Original Urdu Text, with tags:
صحبت ہے عجب گرم ہے <LOCATION>دربار حسینی</LOCATION>
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
2. Entity: دربار حسینی
Predicted NER Tag: LOCATION

Original Urdu Text, with tags:
سب بزم ہے مشتاق <PERSON>علم دار حسینی</PERSON>
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
3. Entity: علم دار حسینی
Predicted NER Tag: PERSON

Original Urdu Text, with tags:
<PERSON>عباس علی</PERSON> <PERSON>اختر</PERSON> <PERSON>اقبال علی</PERSON> ہے
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
4. Entity: عباس علی
Predicted NER Tag: PERSON
5. Entity: اختر
Predicted NER Tag: PERSON
6. Entity: اقبال علی
Predicted NER Tag: PERSON

Original Urdu Text, with tags:
شوکت سے عیاں <DESIGNATION>حشمت</DESIGNATION> <DESIGNATION>واجلال</DESIGNATION> علی ہے
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
7. Entity: حشمت
Predicted NER Tag: DESIGNATION
8. Entity: واجلال
Predicted NER Tag: DESIGNATION

Original Urdu Text, with tags:
خاتم یہ جہاں کے نہیں <LOCATION>درِّ نجف</LOCATION> ایسا
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
9. Entity: درِّ نجف
Predicted NER Tag: LOCATION

Original Urdu Text, with tags:
سیاف غزا <DESIGNATION>سردار</DESIGNATION> و <DESIGNATION>جرار</DESIGNATION>
Context:
پتلی کی طرح نظر سے مستور ہے تو
آنکھیں جسے ڈھونڈھتی ہیں وہ نور ہے تو
ہے قریب رگ جان سے اس پر یہ بعُد
اللہ اللہ کس قدر دور ہے تو

Extracted Entities:
10. Entity: سردار
Predicted NER Tag: DESIGNATION
11. Entity: جرار
Predicted NER Tag: DESIGNATION

---END OF SENTENCES---

Output:

{
  "predictions": [
    {
      "entity": "میر انیس",
      "tag": "PERSON",
      "correct": true,
      "alternative": "PERSON"
    },
    {
      "entity": "دربار حسینی",
      "tag": "LOCATION",
      "correct": true,
      "alternative": "LOCATION"
    },
    {
      "entity": "علم دار حسینی",
      "tag": "PERSON",
      "correct": false,
      "alternative": "DESIGNATION"
    },
    {
      "entity": "عباس علی",
      "tag": "PERSON",
      "correct": true,
      "alternative": "PERSON"
    },
    {
      "entity": "اختر",
      "tag": "PERSON",
      "correct": true,
      "alternative": "PERSON"
    },
    {
      "entity": "اقبال علی",
      "tag": "PERSON",
      "correct": true,
      "alternative": "PERSON"
    },
    {
      "entity": "حشمت",
      "tag": "DESIGNATION",
      "correct": false,
      "alternative": "PERSON"
    },
    {
      "entity": "واجلال",
      "tag": "DESIGNATION",
      "correct": false,
      "alternative": "PERSON"
    },
    {
      "entity": "درِّ نجف",
      "tag": "LOCATION",
      "correct": true,
      "alternative": "LOCATION"
    },
    {
      "entity": "سردار",
      "tag": "DESIGNATION",
      "correct": true,
      "alternative": "DESIGNATION"
    },
    {
      "entity": "جرار",
      "tag": "DESIGNATION",
      "correct": false,
      "alternative": "PERSON"
    },
    {
      "entity": "اولو العزم",
      "tag": "ORGANIZATION",
      "correct": false,
      "alternative": "PERSON"
    },
    {
      "entity": "محبوب الٰی",
      "tag": "PERSON",
      "correct": false,
      "alternative": "DESIGNATION"
    },
    {
      "entity": "شہ",
      "tag": "DESIGNATION",
      "correct": false,
      "alternative": "PERSON"
    }
  ]
}

"""

NER_USER_PROMPT = """

---BEGINNING OF SENTENCES---

{sentences}

---END OF SENTENCES---
"""


def get_evaluation_data(
  tagged_data, 
  sentence_chunk_size=SENTENCES_CHUNK, 
  context_size=CONTEXT_LENGTH,
  system_prompt=NER_JUDGE_SYSTEM_PROMPT_REFINED,
):
    def build_sentences_prompt(tagged_sentences):
        ner_prompt = ""
        count = 0
        for _, s in enumerate(tagged_sentences):
            ner_prompt += "Original Urdu Text, with tags:\n"
            ner_prompt += f"{s['tagged']}\n"
            ner_prompt += f"Context:\n{s['context']}\n\n"
            ner_prompt += "Extracted Entities:\n"
            
            for entity in s['entities']:
                count += 1
                ner_prompt += f"{count}. Entity: {entity['entity']}\n"
                ner_prompt += f"Predicted NER Tag: {entity['tag']}\n"
            ner_prompt += "\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": NER_USER_PROMPT.format(sentences=ner_prompt)}
        ]
        
        return messages
    
    all_sentences_data = list()
    for i, d in enumerate(tagged_data):
        context = "\n".join([
            i['original'] 
            for i in tagged_data[max(0, i-context_size):min(len(tagged_data), i+context_size)]
        ])
        original = d['original']
        tagged = d['tagged']
        entities = [v for k, v in d['entity_status'].items() if k != 'user_verified' and len(v) > 0]
        if not entities:
            continue
        all_sentences_data.append({
            'context': context,
            'original': original,
            'tagged': tagged,
            'entities': entities
        })


    messages_chunks = list()

    for i in range(0, len(all_sentences_data), sentence_chunk_size):
        sentences = all_sentences_data[i:i+sentence_chunk_size]
        messages_chunks.append(build_sentences_prompt(sentences))

    return messages_chunks


def query_llms(messages: List[Dict[str, str]], llm_names: List[str]) -> List[str]:
    responses = dict()
    for llm in list([LLM(llm_name, response_format=LLMJudgement) for llm_name in llm_names]):
        print(f"Querying {llm.model}...")
        try:
            start_time = time.time()
            resp = llm.call(messages)
            print(f"Response time for {llm.model}: {time.time() - start_time:.2f} seconds")
            responses[llm.model] = format_llm_response(resp)
        except Exception as e:
            print(f"Error querying {llm.model}: {e}")
    
    return responses


def judge_message_chunks(all_message_chunks: List[List[Dict[str, str]]], llm_names: List[str], tqdm = tqdm) -> Dict[str, List[str]]:
    """
    Judge the messages using the LLM and return the responses.
    
    Args:
        all_message_chunks (List[List[Dict[str, str]]]): List of message chunks for judgement.
    
    Returns:
        Dict[str, List[str]]: Dictionary with LLM names as keys and their respective responses as values.
    """
    extracted_results = [None] * len(all_message_chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(query_llms, chunk, llm_names): idx
            for idx, chunk in enumerate(all_message_chunks)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Judging NER Chunks by LLMs",
        ):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    extracted_results[idx] = result
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
                
    extracted_results = [res for res in extracted_results if res is not None]
    
    return extracted_results


def run_evaluation(
    data, 
    llm_names, 
    sentence_chunk_size=SENTENCES_CHUNK, 
    context_size=CONTEXT_LENGTH,
    tqdm=tqdm
) -> list:

    all_message_chunks = get_evaluation_data(data, sentence_chunk_size, context_size)
    print("Total chunks:", len(all_message_chunks))
    results = judge_message_chunks(all_message_chunks, llm_names, tqdm=tqdm)
    return results
  