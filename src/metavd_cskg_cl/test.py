import pandas as pd
from dotenv import load_dotenv
from numberbatch2 import find_similar_entities
from openai import OpenAI
from settings import CHARADES_CLASSES_CSV, CHARADES_TEST_CSV
from text_to_uri import standardized_uri

load_dotenv()

charades_test_df = pd.read_csv(CHARADES_TEST_CSV)
charades_classes_df = pd.read_csv(CHARADES_CLASSES_CSV)


charades_action_label_list = charades_classes_df.iloc[:, 1].tolist()
charades_caption_list = charades_test_df.iloc[:, 8].tolist()
charades_action_label_id_list = charades_test_df.iloc[:, 9].str.split().str[0].str[1:].apply(lambda x: int(x)).tolist()

system_prompt = (
    "あなたは動画のキャプション中の動作ラベルをConceptNet上の対応するエンティティとリンクするエンティティリンカーです"
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


for charades_caption, charades_action_label_id in zip(
    charades_caption_list, charades_action_label_id_list, strict=False
):
    action_label = standardized_uri("en", charades_action_label_list[charades_action_label_id]).split("/")[-1]
    print(action_label)
    similar_entities = find_similar_entities(action_label)
    print(similar_entities)
    # prompt = (
    #     "あなたには動画のキャプションとその動画の動作ラベルが与えられます。\n"
    #     "行動ラベルを基にキャプションから重要なキーワードを抽出してください。\n"
    #     f"動画のキャプション: {charades_caption}\n"
    #     f"動画の動作ラベル: {action_label}\n"
    # )
    # print(completioin(prompt))
