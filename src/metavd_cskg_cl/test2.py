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

for i, charades_caption in enumerate(charades_caption_list, start=1):
    print(i, charades_caption)

system_prompt = (
    "あなたは動画の動作ラベルをConceptNet上の関連するエンティティとリンクするエンティティリンキングツールです"
)

client = OpenAI()


def completioin(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


# with CHARADES_RESULT_JSON.open("w") as f:
#     f.write("[\n")
#     for idx, (charades_caption, charades_action_label_id) in enumerate(
#         zip(charades_caption_list, charades_action_label_id_list, strict=True), start=1
#     ):
#         action_label = standardized_uri("en", charades_action_label_list[charades_action_label_id]).split("/")[-1]
#         # print(action_label)
#         # similar_entities = find_similar_entities(action_label)
#         # print(similar_entities)
#         prompt = (
#             "あなたにはある動画の動作ラベルが与えられます。\n"
#             "動作ラベルと関連するConceptNet上のエンティティすべてを出力してください。\n"
#             "また、出力は以下のフォーマットを遵守し、それ以外の出力は絶対にしないでください。\n"
#             '"entitiy": [エンティティ1, エンティティ2, ...]\n\n'
#             # f"動画のキャプション: {charades_caption}\n"
#             f"動画の動作ラベル: {action_label}\n"
#         )
#         f.write(f'{{"id": {idx}, {completioin(prompt)}}},\n')
#     f.write("]\n")
