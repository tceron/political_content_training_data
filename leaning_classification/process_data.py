import os
import re
import pandas as pd
import glob


def process_news_articles_dataset():
    df = pd.read_csv('data/news_articles_leaning.csv')
    df.insert(0, "prompt_id", df.index)

    slant2class = {4:"neutral",  3.5:"neutral", 3:"left",  4.5:"neutral", 2.5:"left", 5:"right",  
                1.5:"left", 2:"left",  6:"right",  6.5: "right", 7: "right",  5.5: "right", 1: "left"}

    df['label'] = df['slantlabel_final'].map(slant2class)
    df_final = df.copy()
    # testset = pd.read_csv("data/slant_testset.csv")
    # testset = pd.concat([testset, pd.read_csv("data/slant_train.csv")], axis=1)

    # df_final = pd.DataFrame()
    # for label in df.label.unique():
    #     tmp = df[df.label == label]
    #     tmp = tmp[~tmp['prompt_id'].isin(testset['prompt_id'])]
    #     tmp = tmp.sample(n=4, random_state=42, replace=False)
    #     df_final = pd.concat([df_final, tmp])
    df_final = df_final.reset_index(drop=True)
    df_final['content'] = df_final['title'] + ' ' + df_final['text.onelinersRemoved']
    df_final = df_final[["prompt_id", 'content', 'label', "slantlabel_final"]]
    df_final["label_code"]= df_final['label'].map({'left': 1, 'neutral': 2, 'right': 3})
    df_final.to_csv("data/news_articles_leaning_all.csv", index=False)
    print(df_final)


# df = pd.read_csv("data/news_articles_leaning.csv")
# urls = [i.split("/")[0] for i in df['canonicalurl']]
# print(len(set(urls)))

def create_stratified_samples_new_leaning_dataset(n_samples=5):
    df = pd.read_csv("/home/ceron/news_article_slant_classification/data/news_articles_leaning.csv")

    df.insert(0, "prompt_id", df.index)

    slant2class = {4:"neutral",  3.5:"neutral",  4.5:"neutral",
                   2.5:"left",  3:"left",  1: "left", 1.5:"left", 2:"left",  
                   6:"right",   5:"right",  5.5: "right",
                   6.5: "right", 7: "right"}

    df['label'] = df['slantlabel_final'].map(slant2class)

    df = df.reset_index(drop=True)
    df['content'] = df['title'] + ' ' + df['text.onelinersRemoved']
    df = df[["prompt_id", 'content', 'label', "slantlabel_final"]]
    df["label_code"]= df['label'].map({'left': 1, 'neutral': 2, 'right': 3})

    for n in range(n_samples):
        tmp = df.groupby('label_code', group_keys=False).apply(lambda g: g.sample(300, random_state=n))
        tmp.to_csv(f"data/new_leaning_sample_{n}.csv", index=False)
        print(f"Sample {n} created with {len(tmp)} rows.")
        print(tmp.label.value_counts())
        print(tmp)

# create_stratified_samples_new_leaning_dataset(n_samples=5)

def subsample_neutral():
    df = pd.read_csv("/home/ceron/news_article_slant_classification/data/news_articles_leaning.csv")

    df.insert(0, "prompt_id", df.index)

    slant2class = {4:"neutral",  3.5:"neutral",  4.5:"neutral",
                   2.5:"left",  3:"left",  1: "left", 1.5:"left", 2:"left",  
                   6:"right",   5:"right",  5.5: "right",
                   6.5: "right", 7: "right"}

    df['label'] = df['slantlabel_final'].map(slant2class)
    neutral = df[df['label'] == 'neutral']
    neutral = neutral.sample(n=1148, random_state=42)
    not_neutral = df[df['label'] != 'neutral']
    df = pd.concat([neutral, not_neutral])

    df = df.reset_index(drop=True)
    df['content'] = df['title'] + ' ' + df['text.onelinersRemoved']
    df = df[["prompt_id", 'content', 'label', "slantlabel_final"]]
    df["label_code"]= df['label'].map({'left': 1, 'neutral': 2, 'right': 3})
    print(df)
    print(df.label.value_counts())
    df.to_csv("data/news_articles_leaning_subsampled_neutral.csv", index=False)

# subsample_neutral()

def gather_classifier_subsampled_gpt():
    classified_files = glob.glob("results_gpt/gpt*.csv")
    reference_df = pd.read_csv("data/news_articles_leaning_subsampled_neutral.csv")
    ids_to_keep = reference_df['prompt_id'].tolist()

    groups = {}
    for file in classified_files:
        df = pd.read_csv(file)
        if 'id' not in df.columns:
            continue
        model, prompt_template = os.path.basename(file)[:-len(".csv")].split('_', 1)
        prompt_template = re.sub(r'_new_leaning_sample_\d+$', '', prompt_template)
        groups.setdefault((model, prompt_template), []).append(df)

    for (model, prompt_template), dfs in groups.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined[combined['id'].isin(ids_to_keep)]
        combined = combined.drop_duplicates(subset='id')
        combined.to_csv(f"results_gpt/{model}_{prompt_template}_subsample.csv", index=False)

# gather_classifier_subsampled_gpt()

# df = pd.read_csv("results_gpt/gpt-5.4-mini_zero_2_subsample.csv")
# print(df.gold.value_counts())

def join_files():
    """ Join files ending in _subsample.csv and _missing.csv into a single file for each model and prompt template inside results_gpt/. """
    files = glob.glob("results_gpt/*_subsample.csv") + glob.glob("results_gpt/*_missing.csv")

    groups = {}
    for file in files:
        base = os.path.basename(file)[:-len(".csv")]
        for suffix in ("_subsample", "_missing"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        groups.setdefault(base, []).append(file)

    for base, group_files in groups.items():
        combined = pd.concat([pd.read_csv(f) for f in group_files], ignore_index=True)
        combined = combined.drop_duplicates(subset='id')
        combined.to_csv(f"results_gpt/{base}_full.csv", index=False)
        print(f"{base}_full.csv: {len(combined)} rows from {len(group_files)} file(s)")

# join_files()

