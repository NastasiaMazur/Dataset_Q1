from openai import OpenAI
import json
import os
import requests



client = OpenAI(
    api_key="",
)


def get_language_links(title, lang="en"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "langlinks",
        "lllimit": "max",
        "redirects": 1,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("langlinks", [])
    return []


def fetch_wikipedia_article(title, lang="en"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "Article not found")
    return "Error fetching article"


def get_first_paragraph(text):
    return text.split("\n")[0]


def generate_query(positive_passage, lang):
    prompt = (
        f"You are an annotator for creating a custom CLIR dataset similar to mMarco. "
        f"Based on the following text from a psychology article, generate a specific and relevant query in {lang} "
        f"that could be used to search for key information in the article. Make sure the query captures the main idea "
        f"of the passage: {positive_passage}. The query must be in {lang}."
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model="gpt-4o-mini"
    )
    query = response.choices[0].message.content.strip()
    return query


def generate_negative_passage(query, lang):
    prompt = (
        f"Translate the query ‘{query}’ into {lang}. Find an existing passage in the internet/journal/book in {lang}, "
        f"that includes at least one of the identified topic's keywords from the query. The passage must not answer the query directly but must contain keywords. The passage must be unrelated to the psychological domain, "
        f"originating from fields such as politics, economics, management, technology, or other. Do not show me the keywords, do not show me the translation, "
        f"no resources, do not include extra text. Do not tell me ‘Here is ...’ or similar! Provide the passage text in {lang} ONLY!. JUST SHOW ME THE ARTICLE'S TEXT!"
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model="gpt-4o-mini"
    )
    negative_passage = response.choices[0].message.content.strip()
    return negative_passage


def translate_passage(passage, target_lang):
    prompt = (
        f"You are a professional translator in the psychological domain. Please translate this text into {target_lang}, "
        f"taking care of cultural and language nuances as well as the style of the target language. "
        f"The target readers are professional psychologists, so please use the correct terminology. Passage to translate: {passage}"
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model="gpt-4o-mini"
    )
    translated_passage = response.choices[0].message.content.strip()
    return translated_passage


def create_schema(titles):
    schema = []
    failed_titles = []

    languages = ["en", "de", "ru"]

    for title in titles:
        print(f"Processing title: {title}")
        language_links = get_language_links(title)
        original_passages = {}
        translated_passages = {}

        for lang in languages:
            if lang == "en":
                positive_title = title
            else:
                positive_title = next(
                    (link["*"] for link in language_links if link["lang"] == lang), None
                )

            if positive_title:  # Does the title exist?
                positive_passage = fetch_wikipedia_article(positive_title, lang)
                positive_passage_first = get_first_paragraph(positive_passage)
                original_passages[lang] = positive_passage_first

                translated_passages[lang] = {}
                for other_lang in languages:
                    if other_lang != lang:
                        translated_passages[lang][other_lang] = translate_passage(
                            positive_passage_first, other_lang
                        )
            else:
                print(f"Failed to find article for {title} in {lang}")
                failed_titles.append(title)

        if len(original_passages) == 3:
            combinations = [
                {"query_lang": "en", "pos_lang": "de", "neg_lang": "en"},
                {"query_lang": "en", "pos_lang": "ru", "neg_lang": "en"},
                {"query_lang": "en", "pos_lang": "de", "neg_lang": "ru"},
                {"query_lang": "en", "pos_lang": "ru", "neg_lang": "de"},
                {"query_lang": "de", "pos_lang": "en", "neg_lang": "de"},
                {"query_lang": "de", "pos_lang": "ru", "neg_lang": "de"},
                {"query_lang": "de", "pos_lang": "en", "neg_lang": "ru"},
                {"query_lang": "de", "pos_lang": "ru", "neg_lang": "en"},
                {"query_lang": "ru", "pos_lang": "en", "neg_lang": "ru"},
                {"query_lang": "ru", "pos_lang": "de", "neg_lang": "ru"},
                {"query_lang": "ru", "pos_lang": "en", "neg_lang": "de"},
                {"query_lang": "ru", "pos_lang": "de", "neg_lang": "en"},
            ]

            for comb in combinations:
                query_lang = comb["query_lang"]
                pos_lang = comb["pos_lang"]
                neg_lang = comb["neg_lang"]

                if pos_lang in original_passages:
                    positive_passage = original_passages[pos_lang]
                else:
                    positive_passage = translated_passages[pos_lang][query_lang]

                schema.append(
                    {
                        "query": generate_query(positive_passage, query_lang),
                        "positive_passage": positive_passage,
                        "negative_passage": generate_negative_passage(
                            positive_passage, neg_lang
                        ),
                        "query_language": query_lang,
                        "positive_language": pos_lang,
                        "negative_language": neg_lang,
                        "q_type": "1",
                    }
                )

    return schema, failed_titles


def save_schema_to_json(schema, filename="dataset.json"):

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(schema)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


def save_failed_titles(failed_titles, filename="failed_titles.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(failed_titles, f, ensure_ascii=False, indent=4)


titles = [
    "Impostor_syndrome",
    "Anxiety",
    "Anxiety_disorder",
    "Emotion",
    "Shame",
    "Panic_disorder",
    "Panic attack",
    "Psychological_resilience",
    "Social_identity_theory",
    "Big_Five_personality_traits",
    "Identity_(social_science)",
    "Cognitive_bias",
    "Cognitive_behavioral_therapy",
    "Well-being_contributing_factors",
    "Neuroplasticity",
    "Emotional_intelligence",
    "Self-categorization_theory",
    "Group_dynamics",
    "Collective_identity",
    "Another_Title",
]  # TODO: add more


schema_results, failed_titles = create_schema(titles)

save_schema_to_json(schema_results, "dataset_q_type_1.json")

if failed_titles:
    save_failed_titles(failed_titles, "failed_titles.json")

print(f"Total samples written: {len(schema_results)}.")
print(f"Failed to find articles for {len(failed_titles)} titles.")
