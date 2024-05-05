import sys
from os import cpu_count
from utils import augument_text, calculate_embedding_similarity, augument_sentence
from utils import seed_everything, lemmatization_and_tokenization, calculate_imbalance_ratio, plot_history, plot_confusion_matrix, plot_precision_recall_curve, plot_ROC_curve
import pandas as pd
from tqdm import tqdm

# DL imports
from train_and_eval import train, get_predictions
from dataset import TextDataset
import torch
import transformers
from torch.utils.data import DataLoader
import torch.optim as optim
from model import BERTClassifier

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score

import spacy
nlp = spacy.load("it_core_news_sm")
tqdm.pandas()


sys.path.append('src/')


N_CORES = cpu_count()
print(f'Number of Logical CPU cores: {N_CORES}')

base_models_mapping = {
    'distilbert': 'distilbert-base-uncased',
    'bertino': 'indigo-ai/BERTino',
    'electraita': 'dbmdz/electra-base-italian-xxl-cased-discriminator',
    'bertita': 'dbmdz/bert-base-italian-xxl-uncased'
}

DATA_PATH = "/home/user/Documents/Dataset/dataset/second_training/testing_aug"
AUG_METHOD = "EDA"  # or "RAW" or "BT" or "CWEW" or "EDA"
LANGUAGE = "italian"  # or "english" or "italian-legal"
# "DL" or "ML_NB" or "ML_SVM" or "ML_LR" or "ML_RF" or "ML_Ensemble"
CLASSIFICATION_METHOD = "DL"
TKNZ_AND_LEMM = False
BASE_MODEL_NAME = 'bert'
BERT_FINETUNED_NAME_CLEAN = 'bertita'
BERT_FINETUNED_NAME = base_models_mapping[BERT_FINETUNED_NAME_CLEAN]
MAX_LEN = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5  # 2e-5, 3e-5, 5e-5
EPOCHS = 30
NUM_CLASSES = 23
# PATIENCE_ES = 5, not used
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
SEEDS = [42]  # 1, 42, 123
N_RUN = 1  # 3
if CLASSIFICATION_METHOD == "DL":
    MODEL_PATH = DATA_PATH + "/" + BERT_FINETUNED_NAME_CLEAN
else:
    MODEL_PATH = DATA_PATH + "/ML"
print(f"Data path: {DATA_PATH}")
print(f"Model path: {MODEL_PATH}")


def main():
    avg_testacc = 0.0
    avg_testf1 = 0.0

    for seed, run in zip(SEEDS, range(N_RUN)):
        seed_everything(seed)
        print(f"Actual SEED: {torch.initial_seed()}")

        print("Tokenization and lemmatization: {}".format(TKNZ_AND_LEMM))

        try:
            train_df = pd.read_csv(
                f"{DATA_PATH}/train_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\")
            # fill nan values with empty string
            train_df.fillna("", inplace=True)
            train_df = train_df.dropna()

            test_df = pd.read_csv(
                f"{DATA_PATH}/test_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\")
            # fill nan values with empty string
            test_df.fillna("", inplace=True)
            test_df = test_df.dropna()

            val_df = pd.read_csv(
                f"{DATA_PATH}/val_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\")
            val_df.fillna("", inplace=True)
            val_df = val_df.dropna()

            print("Train, Test and Val DF loaded! Method: {} and tokenization&lemmatiz:{}".format(
                AUG_METHOD, TKNZ_AND_LEMM))
        except:
            print("Train, Test and Val DF not found, loading dataframe...")
            # load data
            df = pd.read_csv(f"{DATA_PATH}/all_data_cleaned.csv",
                             sep="\t", escapechar="\\")

            print("Dataframe loaded!")
            df['year'] = df['year'].fillna("")
            df['id'] = df['id'].fillna("")
            df['main_text'] = df['main_text'].fillna("")
            df['subject'] = df['subject'].fillna("")
            df['attachs'] = df['attachs'].fillna("")
            df['from'] = df['from'].fillna("")
            df['cc'] = df['cc'].fillna("")
            df['filenames'] = df['filenames'].fillna("")
            df['size_attachs'] = df['size_attachs'].fillna("")

            df = df.dropna()

            # low-resources scenario: 5%, 10%, 20%, 60% of the dataset
            # stratified split
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.4, random_state=seed)
            for train_index, test_index in sss.split(df, df['label']):
                train_df, test_df = df.iloc[train_index], df.iloc[test_index]

            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=seed)
            for test_index, val_index in sss2.split(test_df, test_df['label']):
                test_df, val_df = test_df.iloc[test_index], test_df.iloc[val_index]

            print("Train shape: {}, Val shape: {}, Test shape: {}.".format(
                train_df.shape, val_df.shape, test_df.shape))

            print("Complete Dataframe values distribution on year:",
                  df['year'].value_counts())
            # check if the split is stratified
            print("Train Dataframe values distribution on year:",
                  train_df['year'].value_counts())
            print("Test Dataframe values distribution on year:",
                  test_df['year'].value_counts())
            print("Val Dataframe values distribution on year:",
                  val_df['year'].value_counts())

            print("Complete Dataframe values distribution on label:",
                  df['label'].value_counts())
            print("Train Dataframe values distribution on label:",
                  train_df['label'].value_counts())
            print("Test Dataframe values distribution on label:",
                  test_df['label'].value_counts())
            print("Val Dataframe values distribution on label:",
                  val_df['label'].value_counts())

            if AUG_METHOD != "RAW":
                print("*"*50)
                imbalanced_ratios = calculate_imbalance_ratio(df)
                print("Imbalanced ratios on the complete dataframe:",
                      imbalanced_ratios)
                print("*"*50)

                # COSINE EMBEDDING SIMILARITY AND BLEU SCORE CALCULATION:
                # if run == 0:
                #     # calculate the embedding similarity between the original text and the augmented text for the first run
                #     text_to_aug = train_df[train_df['label']
                #                            == 21]['text'].values.tolist()

                #     for text in text_to_aug:
                #         if type(text) == float:  # for nan values
                #             text_to_aug.remove(text)

                #     text_augmented = []

                #     for text in text_to_aug:
                #         for aug_text in augument_sentence(text, lang=LANGUAGE, method=AUG_METHOD, num_aug=1, n_thread=N_CORES):
                #             text_augmented.append(aug_text)

                #     embedding_similarity = calculate_embedding_similarity(
                #         text_to_aug, text_augmented)
                #     embedding_similarity = round(embedding_similarity, 2)
                #     print("Embedding similrity for the method {} is: {}".format(
                #         AUG_METHOD, embedding_similarity))

                for label, ratio in imbalanced_ratios.items():
                    if label == 8:
                        print("Augumenting label: {}".format(label))
                        train_df = augument_text(
                            df=train_df, label_to_aug=label, method=AUG_METHOD, lang=LANGUAGE, num_aug=3, n_thread=N_CORES)
                    if ratio >= 9.9 and ratio < 19.9:
                        print("Augumenting label: {}".format(label))
                        if label == 18 or label == 21:
                            train_df = augument_text(
                                df=train_df, label_to_aug=label, method=AUG_METHOD, lang=LANGUAGE, num_aug=3, n_thread=N_CORES, bert_type="legal")
                        else:
                            train_df = augument_text(
                                df=train_df, label_to_aug=label, method=AUG_METHOD, lang=LANGUAGE, num_aug=3, n_thread=N_CORES)
                    elif ratio >= 19.9:
                        print("Augumenting label: {}".format(label))
                        if label == 16:
                            train_df = augument_text(
                                df=train_df, label_to_aug=label, method=AUG_METHOD, lang="english", num_aug=4, n_thread=N_CORES)
                        else:
                            train_df = augument_text(
                                df=train_df, label_to_aug=label, method=AUG_METHOD, lang=LANGUAGE, num_aug=4, n_thread=N_CORES)
                    else:
                        print("No augumentation for label: {}".format(label))

                    print("Augumented train data shape: ", train_df.shape)

            if TKNZ_AND_LEMM:
                print("Tokenized dataframe not found, tokenizing...")

                train_df["text"] = train_df["from"] + " " + train_df["cc"] + " " + train_df["subject"] + \
                    " " + train_df["main_text"] + " " + \
                    train_df["filenames"] + " " + train_df["attachs"]
                train_df["text_len"] = train_df["text"].apply(lambda x: len(x))

                test_df["text"] = test_df["from"] + " " + test_df["cc"] + " " + test_df["subject"] + \
                    " " + test_df["main_text"] + " " + \
                    test_df["filenames"] + " " + test_df["attachs"]
                test_df["text_len"] = test_df["text"].apply(lambda x: len(x))

                val_df["text"] = val_df["from"] + " " + val_df["cc"] + " " + val_df["subject"] + \
                    " " + val_df["main_text"] + " " + \
                    val_df["filenames"] + " " + val_df["attachs"]
                val_df["text_len"] = val_df["text"].apply(lambda x: len(x))

                nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])

                lemmatized_and_tokenized_text_train = lemmatization_and_tokenization(
                    train_df["text"], nlp, train_df["text_len"].max())
                lemmatized_and_tokenized_text_test = lemmatization_and_tokenization(
                    test_df["text"], nlp, test_df["text_len"].max())
                lemmatized_and_tokenized_text_val = lemmatization_and_tokenization(
                    val_df["text"], nlp, val_df["text_len"].max())

                # tokenize the dataset with spacy and progress apply
                train_df['text'] = lemmatized_and_tokenized_text_train
                test_df['text'] = lemmatized_and_tokenized_text_test
                val_df['text'] = lemmatized_and_tokenized_text_val
                print("Tokenization done!")

            # writ the augmented dataframe to csv
            train_df.to_csv(
                f"{DATA_PATH}/train_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\", index=False)
            test_df.to_csv(
                f"{DATA_PATH}/test_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\", index=False)
            val_df.to_csv(
                f"{DATA_PATH}/val_data_aug_{AUG_METHOD}_token{TKNZ_AND_LEMM}.csv", sep="\t", escapechar="\\", index=False)

            print(f"Train, Test and Val DF saved! {AUG_METHOD}")

        if CLASSIFICATION_METHOD == "DL":
            # use <SEP> token to separate the text fields
            train_df['text'] = train_df['from'] + " <SEP> " + train_df['cc'] + " <SEP> " + train_df['subject'] + \
                " <SEP> " + train_df['main_text'] + " <SEP> " + \
                train_df['filenames'] + " <SEP> " + train_df['attachs']

            test_df['text'] = test_df['from'] + " <SEP> " + test_df['cc'] + " <SEP> " + test_df['subject'] + \
                " <SEP> " + test_df['main_text'] + " <SEP> " + \
                test_df['filenames'] + " <SEP> " + test_df['attachs']

            val_df['text'] = val_df['from'] + " <SEP> " + val_df['cc'] + " <SEP> " + val_df['subject'] + \
                " <SEP> " + val_df['main_text'] + " <SEP> " + \
                val_df['filenames'] + " <SEP> " + val_df['attachs']

            tokenizer = None

            if BASE_MODEL_NAME == 'distilbert':
                tokenizer = transformers.DistilBertTokenizer.from_pretrained(
                    BERT_FINETUNED_NAME)
            elif BASE_MODEL_NAME == 'electra':
                tokenizer = transformers.ElectraTokenizer.from_pretrained(
                    BERT_FINETUNED_NAME)
            elif BASE_MODEL_NAME == 'bert':
                tokenizer = transformers.BertTokenizer.from_pretrained(
                    BERT_FINETUNED_NAME)

            print(f"MAX LEN: {MAX_LEN}")

            model = BERTClassifier(BERT_FINETUNED_NAME, NUM_CLASSES, BASE_MODEL_NAME).to(
                device)  # create model

            try:
                model.load_state_dict(torch.load(
                    f"{MODEL_PATH}/best_model_state.bin"))
                print("Pretrained model loaded, further training!")
            except:
                print("Pretrained model not found, training from scratch!")

            # create dataset and dataloader
            train_data = TextDataset(text=train_df.text.to_numpy(),
                                     target=train_df.label.to_numpy(),
                                     tokenizer=tokenizer,
                                     max_len=MAX_LEN)

            # test dataset
            test_data = TextDataset(text=test_df.text.to_numpy(),
                                    target=test_df.label.to_numpy(),
                                    tokenizer=tokenizer,
                                    max_len=MAX_LEN)
            # test data loader
            val_data = TextDataset(text=val_df.text.to_numpy(),
                                   target=val_df.label.to_numpy(),
                                   tokenizer=tokenizer,
                                   max_len=MAX_LEN)

            train_loader = DataLoader(
                train_data, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(
                val_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(
                test_data, batch_size=BATCH_SIZE, shuffle=True)

            # # freeze BERT parameters
            # for param in model.bert.embeddings.parameters():
            #     param.requires_grad = False

            criterian = torch.nn.CrossEntropyLoss().to(device)
            opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            # start training
            print(f"Training for run {run}")
            start_time = pd.Timestamp.now()
            print("--------------------------------------------")
            history, _, best_model_wts = train(
                model, train_loader, val_loader, criterian, opt, device, EPOCHS, MODEL_PATH)
            print(f"Terminating training for run {run}")
            finish_time = pd.Timestamp.now()
            print("Training time: ", finish_time - start_time)
            print("--------------------------------------------")
            plot_history(history, run)

            model.load_state_dict(best_model_wts)

            _, y_pred, y_pred_probs, y_test = get_predictions(
                model, test_loader, device)

            # plot confusion matrix
            plot_confusion_matrix(y_pred.cpu(), y_test.cpu())
            # plot ROC and Precision-Recall curves
            plot_ROC_curve(y_pred.cpu(), y_test.cpu(),
                           NUM_CLASSES, proba=False)
            plot_precision_recall_curve(
                y_pred.cpu(), y_test.cpu(), NUM_CLASSES, proba=False)

        else:
            tfidf_vect = TfidfVectorizer(max_features=5000)
            tfidf_vect.fit(train_df['text'].values.tolist() +
                           test_df['text'].values.tolist())

            x_train_tfidf = tfidf_vect.transform(
                train_df['text'].values.tolist())
            x_test_tfidf = tfidf_vect.transform(
                test_df['text'].values.tolist())

            if CLASSIFICATION_METHOD == "ML_SVM":
                ML_model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
            elif CLASSIFICATION_METHOD == "ML_NB":
                ML_model = MultinomialNB()
            elif CLASSIFICATION_METHOD == "ML_LR":
                ML_model = LogisticRegression(random_state=seed)
            elif CLASSIFICATION_METHOD == "ML_RF":
                ML_model = RandomForestClassifier(
                    n_estimators=100, random_state=seed)
            elif CLASSIFICATION_METHOD == "ML_Ensemble":
                ML_model = VotingClassifier(
                    estimators=[('lr', LogisticRegression(random_state=seed)), ('rf', RandomForestClassifier(n_estimators=100, random_state=seed)), ('svm', SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))], voting='soft')

            print("Fitting ML model...")
            ML_model.fit(x_train_tfidf, train_df['label'].values.tolist())
            print("ML model fitted!")

            y_pred = ML_model.predict(x_test_tfidf)
            y_test = test_df['label'].values.tolist()

            print(classification_report(y_test, y_pred, digits=4))
            plot_confusion_matrix(y_pred, y_test)

            # plot ROC and Precision-Recall curves
            plot_ROC_curve(y_pred, y_test, NUM_CLASSES, proba=False)
            plot_precision_recall_curve(
                y_pred, y_test, NUM_CLASSES, proba=False)

            return

        curtestacc = (y_pred.cpu() == y_test.cpu()
                      ).sum().item()/len(y_test.cpu())
        curtestf1 = f1_score(y_test.cpu(), y_pred.cpu(), average='macro')

        curtestacc = round(curtestacc, 4)
        curtestf1 = round(curtestf1, 4)

        print("Run {}. Test Accuracy: {}. F1: {}".format(
            run, curtestacc, curtestf1))
        print("Run {}. Test Classification report:".format(run))
        print(classification_report(y_test.cpu(), y_pred.cpu(), digits=4))
        print("---------------------------------------------------")
        avg_testacc += curtestacc
        avg_testf1 += curtestf1

    print("With method: {} the average Test Accuracy is: {}. F1: {} ".format(
        AUG_METHOD, avg_testacc/N_RUN, avg_testf1/N_RUN))


if __name__ == "__main__":
    main()
