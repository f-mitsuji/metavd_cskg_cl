import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from settings import CHARADES_CLASSES_CSV, CHARADES_RESULT_JSON, CHARADES_TEST_CSV
from text_to_uri import standardized_uri

load_dotenv()

charades_test_df = pd.read_csv(CHARADES_TEST_CSV)
charades_classes_df = pd.read_csv(CHARADES_CLASSES_CSV)

charades_action_label_list = charades_classes_df.iloc[:, 1].tolist()
charades_caption_list = charades_test_df.iloc[:, 8].tolist()
charades_action_label_id_list = charades_test_df.iloc[:, 9].str.split().str[0].str[1:].apply(lambda x: int(x)).tolist()

# for i, charades_caption in enumerate(charades_caption_list, start=1):
#     print(i, charades_caption)

system_prompt = (
    "You are an AI assistant tasked with linking action labels to closely related nodes on ConceptNet."
    "Your job is to find all closely related concepts for a given action label and output them in a valid JSON format."
    "Each concept should be represented by its URI on ConceptNet,"
    "and the output should not contain placeholder or incomplete entries."
)

# system_prompt = (
#     "You are an AI assistant tasked with linking action labels to related nodes on ConceptNet."
#     "Your job is to find all related concepts for a given action label and output them in a JSON format."
#     "Each concept should be represented by its URI on ConceptNet."
# )

client = OpenAI()


def completioin(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content


with CHARADES_RESULT_JSON.open("w") as f:
    f.write("[\n")
    for idx, (charades_caption, charades_action_label_id) in enumerate(
        zip(charades_caption_list, charades_action_label_id_list, strict=True), start=1
    ):
        action_label = standardized_uri("en", charades_action_label_list[charades_action_label_id]).split("/")[-1]
        # print(action_label)
        # similar_entities = find_similar_entities(action_label)
        # print(similar_entities)
        prompt = (
            f'For the action label "{action_label}", find all closely related concepts on ConceptNet.'
            # f"For the action label {action_label}, find all related concepts on ConceptNet."
            "Represent each concept using its URI and ensure the output is in a valid JSON format."
            "Each action label should be a key, and the value should be a list of related concept URIs."
            "Do not include placeholders or incomplete entries.\n\n"
            "Example output:\n"
            "{\n"
            '  "playing_soccer": [\n'
            '    "/c/en/play_soccer",\n'
            '    "/c/en/play_football",\n'
            '    "/c/en/playing_football",\n'
            '    "/c/en/soccer",\n'
            '    "/c/en/sports",\n'
            # "    ...\n"
            "  ]\n"
            "}"
        )
        print(prompt)
        # f.write(f'{{"id": {idx}, {completioin(prompt)}}},\n')
        # f.write(f"{completioin(prompt)},\n")
    f.write("]\n")
