from ner_annotator.utils import get_llm_judgment_stats

# from dotenv import load_dotenv
# import re
# from ner_annotator.llm_tagger import get_ner_tags, NERMode
# from ner_annotator.utils import calculate_hash
# import os
# import json
# from tqdm.auto import tqdm


# load_dotenv()
# save_dir = "uploads"


# def tag_file(file_path, mode=NERMode.MARSIYA_ADVANCED):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     selected_file = os.path.basename(file_path)
#     content_hash = calculate_hash(text)
#     save_pth = os.path.join(save_dir, f"{content_hash}_{mode.value}.json")
#     if os.path.exists(save_pth):
#         print(f"File {selected_file} already tagged.")
#         tagged_elements = json.load(open(save_pth, 'r', encoding='utf-8'))['tagged_elements']
#         return tagged_elements
    
#     print(f"Tagging file: {selected_file}")
#     tagged_elements = get_ner_tags(
#         text,
#         mode=mode,
#         model_id="openai/gpt-4.1-mini",
#         chunk_size=40,
#     )
#     save_content = {
#         'text': text,
#         'tagged_elements': tagged_elements,
#     }
#     with open(save_pth, 'w', encoding='utf-8') as f:
#         json.dump(save_content, f, ensure_ascii=False, indent=4)
        
#     return tagged_elements


# def extract_entities(tagged_text):
#     pattern = r"<(.*?)>(.*?)</\1>"
#     return [(match[0], match[1]) for match in re.findall(pattern, tagged_text)]


# def get_entities_dict(tagged_elements):
#     ner_tags = dict()
#     for item in tagged_elements:
#         tags = extract_entities(item['tagged'])
#         for tag_text in tags:
#             if tag_text not in ner_tags:
#                 ner_tags[tag_text] = 0
#             ner_tags[tag_text] += 1
            
#     return ner_tags


# def combine_entities_dicts(d1, d2):
#     combined = {}
#     for tag, count in d1.items():
#         combined[tag] = count
        
#     for tag, count in d2.items():
#         if tag in combined:
#             combined[tag] += count
#         else:
#             combined[tag] = count
#     return combined


# def describe_entities(entities_dict):
#     tagged_elements_dict = {}
#     total_count = 0
#     for tagged_text, value in entities_dict.items():
#         total_count += value
#         tag_type, text = tagged_text
#         # print(f"{tag_type}: {text} - {value} times")
#         if tag_type not in tagged_elements_dict:
#             tagged_elements_dict[tag_type] = 0
#         tagged_elements_dict[tag_type] += 1
    
#     total_unique_tags = sum(tagged_elements_dict.values())
#     print(f"Total unique tags: {total_unique_tags}")
#     print(f"Total tags: {total_count}")
#     return total_count, total_unique_tags


# def annotate_dir(directory: str):
#     entities_dict = dict()
#     for filename in tqdm(os.listdir(directory), desc="Tagging files"):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(directory, filename)
#             tagged_elements = tag_file(file_path)
#             entities_dict = combine_entities_dicts(entities_dict, get_entities_dict(tagged_elements))
#             _, total_unique_tags = describe_entities(entities_dict)
#             if total_unique_tags > 10000:
#                 print(f"Stopping early as total unique entities exceeded 2000: {total_unique_tags}")
#                 break


# annotate_dir('dataset/marsiya-all')

# # fp = 'dataset/test_dir/3. AAJ-SHABBIR-PE-KYA-ALAM-E-TANHAI-HAI.pdf.json.txt'
# # tagged_elements = tag_file(fp, mode=NERMode.MARSIYA)
# # tagged_elements_advanced = tag_file(fp, mode=NERMode.MARSIYA_ADVANCED)