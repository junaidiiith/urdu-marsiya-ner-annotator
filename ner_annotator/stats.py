from collections import Counter

def get_stats(data):
    # print("Calculating statistics...")
    # print(data)
    tagged_lines = data['tagged_elements']
    entity_status = [d['entity_status'] for d in tagged_lines]
    total_entities = [v for es in entity_status for k, v in es.items() if k != 'user_verified']
    per_category_count = dict(Counter([t['tag'] for t in total_entities]))

    total_verified = sum(len(es)-1 if 'user_verified' in es else 0 for es in entity_status)
    # print("entity status: ", entity_status)
    return {
        'total_entities': len(total_entities),
        'per_category_count': per_category_count,
        'total_verified': total_verified,
    }