import re
import os
import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer


device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def calculate_imbalance_ratio(df):
    # Get the counts for each class
    class_counts = df['label'].value_counts()

    # Get the majority class count
    majority_class_count = class_counts.max()

    # Calculate the imbalance ratio for each class
    imbalance_ratios = majority_class_count / class_counts

    # Convert the Series to a dictionary
    imbalance_ratios_dict = imbalance_ratios.to_dict()

    return imbalance_ratios_dict


def plot_history(history, run):
    history['train_f1'] = [v for v in history['train_f1']]
    history['val_f1'] = [v for v in history['val_f1']]
    history['train_loss'] = [v for v in history['train_loss']]
    history['val_loss'] = [v for v in history['val_loss']]

    # set plot figure size
    fig, c_ax = plt.subplots(1, 1, figsize=(10, 7))

    # plot the f1 history
    c_ax.plot(history['train_f1'], label='train', color='blue')
    c_ax.plot(history['val_f1'], label='val', color='orange')
    c_ax.legend()
    c_ax.set_title('Training and Validation F1 Score')
    c_ax.set_xlabel('Epoch')
    c_ax.set_ylabel('F1(Weighted)')
    c_ax.grid(True, alpha=0.5)
    plt.show()

    # plot the loss history
    fig, c_ax = plt.subplots(1, 1, figsize=(10, 7))
    c_ax.plot(history['train_loss'], label='train', color='blue')
    c_ax.plot(history['val_loss'], label='val', color='orange')
    c_ax.legend()
    c_ax.set_title('Training and Validation Loss')
    c_ax.set_xlabel('Epoch')
    c_ax.set_ylabel('Loss')
    c_ax.grid(True, alpha=0.5)
    plt.show()


def lemmatization_and_tokenization(text, nlp, nlp_max_length):
    nlp.max_length = nlp_max_length

    lemm_and_token_text = []

    counter = 0

    for doc in nlp.pipe(text):
        counter += 1
        if counter % 5000 == 0:
            print(counter)
        # also do the lowercase
        lemm_and_token_text.append([token.lemma_.lower() for token in doc])

    # concatenate the tokens in a single string
    lemm_and_token_text = [" ".join(tokens) for tokens in lemm_and_token_text]
    return lemm_and_token_text


def remove_disclaimer(text, sbert_model, pattern_disclaimer, nlp_pipe, spacy_max_length):
    nlp_pipe.max_length = spacy_max_length

    # little pre-processing to (try) overcome the problem that the sentence-bert sbert_model has a max length of 128
    pattern_list = re.compile(
        r"\-+|\*+|\#+", flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    text = pattern_list.sub("", text).strip()

    if len(text) > nlp_pipe.max_length:
        return text

    # tokenize the text with spacy
    tokenized_sentences = nlp_pipe(text).sents

    # 1 CUT: if the sentence matches the pattern disclaimer, remove it
    tokenized_sentences = [
        sentence.text for sentence in tokenized_sentences if not pattern_disclaimer.search(sentence.text)]

    too_long_sent = []
    not_too_long_sent = []

    # 2 CUT: if the sentence is too long for the sentence-bert sbert_model, remove it
    for sentence in tokenized_sentences:
        if len(sbert_model[0].tokenizer(sentence, return_attention_mask=False, return_token_type_ids=False).input_ids) >= sbert_model.max_seq_length:
            too_long_sent.append(sentence)
        else:
            not_too_long_sent.append(sentence)

    # for sentence in not_too_long_sent:
    #     if len(sbert_model[0].tokenizer(sentence, return_attention_mask=False, return_token_type_ids=False).input_ids) >= sbert_model.max_seq_length:
    #         print(len(sbert_model[0].tokenizer(sentence, return_attention_mask=False, return_token_type_ids=False).input_ids))

    # if there are no sentences that makes sense to pass to the sbert_model, return the text
    if len(not_too_long_sent) == 0:
        return text

    # Generate embeddings for the sentences
    embeddings = sbert_model.encode(not_too_long_sent)

    # Generate embeddings for the disclaimer
    disclaimer = "The information in this e-mail is confidential and may be legally privileged"
    embeddings_disclaimer = sbert_model.encode([disclaimer])

    cosine_sim = cosine_similarity(embeddings, embeddings_disclaimer)

    # for sentence, sim in zip(not_too_long_sent, cosine_sim):
    #     if sim > 0.35:
    #         print(sentence, sim)

    not_too_long_sent = [
        sentence
        for sentence, sim in zip(not_too_long_sent, cosine_sim)
        if sim < 0.35
    ]

    return " ".join(too_long_sent + not_too_long_sent)


def format_text_for_BT(language_code, text):
    text = ">>{}<< {}".format(language_code, text)
    return text


def perform_translation(text, model, tokenizer, language):
    # Prepare the text data into appropriate format for the model
    formated_text = format_text_for_BT(language, text)

    # Generate translation using model
    translated = model.generate(
        **tokenizer(formated_text, return_tensors="pt", padding=True).to(device))

    # Convert the generated tokens indices back into text
    translated_text = [tokenizer.decode(
        t, skip_special_tokens=True) for t in translated][0]

    # TODO: approfondisci il perchè ci sono questi caratteri
    translated_text = translated_text.replace("♪", "")
    translated_text = translated_text.replace("#", "")
    translated_text = translated_text.strip()
    return translated_text


def calculate_embedding_similarity(orig_sentences: list[str], gener_sentences: list[str]):

    if len(orig_sentences) != len(gener_sentences):
        raise Exception("The two lists must have the same length")

    model = SentenceTransformer(
        'nickprock/sentence-bert-base-italian-xxl-uncased')

    orig_embeddings = model.encode(orig_sentences)
    print("Original embeddings shape: ", orig_embeddings.shape)
    gener_embeddings = model.encode(gener_sentences)
    print("Generated embeddings shape: ", gener_embeddings.shape)

    # calculate the cosine similarity between the original sentences and the generated sentences
    cosine = [cosine_similarity([orig_embeddings[i]], [gener_embeddings[i]])[
        0][0] for i in range(len(orig_sentences))]
    return np.mean(cosine)


def calculate_bleu(orig_sentences: list[str], gener_sentences: list[str]):
    if len(orig_sentences) != len(gener_sentences):
        raise Exception("The two lists must have the same length")
    bleu = [sentence_bleu([orig_sentences[i]], gener_sentences[i])
            for i in range(len(orig_sentences))]
    return np.mean(bleu)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def augument_sentence(text, lang, method, num_aug, n_thread, bert_type):
    """
    Augment a given sentence using specified method.

    Args:
        text (str): The input sentence to be augmented.
        lang (str): The language of the input sentence.
        method (str): The augmentation method to be used. Supported methods are "SR", "EDA", and "CWE".
        num_aug (int): The number of augmented sentences to generate.
        n_thread (int): The number of threads to use for parallel processing.

    Returns:
        list: A list of augmented sentences.
    """
    if method == "SR":  # apposto
        synonym = naw.SynonymAug(aug_src='wordnet', lang=lang, aug_p=0.3)
        augmented_text = synonym.augment(text, n=num_aug, num_thread=n_thread)
        return augmented_text
    if method == "RD":  # apposto
        rd = naw.RandomWordAug(action="delete", aug_p=0.3)
        augmented_text = rd.augment(text, n=num_aug, num_thread=n_thread)
        return augmented_text
    elif method == "EDA":
        if lang == "italian":
            lang = "ita"
        elif lang == "english":
            lang = "eng"

        eda = naf.Sequential([
            naw.SynonymAug(aug_src='wordnet', lang=lang,
                           aug_p=0.1),  # Synonym Replacement
            naw.RandomWordAug(action='swap', aug_p=0.1),  # Random Swap
            naw.RandomWordAug(action='delete', aug_p=0.1),  # Random Deletion
        ])
        augmented_text = eda.augment(text, n=num_aug, num_thread=n_thread)
        return augmented_text
    elif method == "CWEW":  # apposto
        # XLNet # xlm-roberta-base # dbmdz/bert-base-italian-xxl-uncased # indigo-ai/BERTino
        if lang == "italian" and bert_type == "normal":
            cwe = naw.ContextualWordEmbsAug(
                model_path='dbmdz/bert-base-italian-xxl-uncased', aug_min=1, aug_p=0.2, action="insert")
        if lang == "english" and bert_type == "normal":
            cwe = naw.ContextualWordEmbsAug(
                model_path='xlm-roberta-base', aug_min=1, aug_p=0.2, action="insert")
        if lang == "italian" and bert_type == "legal":
            cwe = naw.ContextualWordEmbsAug(
                model_path='dlicari/Italian-Legal-BERT', aug_min=1, aug_p=0.2, action="insert")
        # indigo-ai/BERTino or dlicari/distil-ita-legal-bert
        augmented_text = cwe.augment(text, n=num_aug, num_thread=n_thread)
        return augmented_text
    # elif method == "GPTJ":
    # https://huggingface.co/andreabac3/Fauno-Italian-LLM-7B
    #     from transformers import AutoTokenizer, AutoModelForCausalLM
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    #     model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    #     # TODO: find an ITALIAN version
    #     # # select two random samples from training set
    #     # text1, label1, text2, label2 = get_two_random_samples()
    #     # # create the prompt
    #     # prompt = get_prompt(text1, label1, text2, label2)
    #     # https://blog.ml6.eu/data-augmentation-using-few-shot-prompting-on-large-language-models-5bc41234d025
    elif method == "BT":  # apposto
        # TODO: single thread, needed multi-thread
        # problema con nlpaug: https://stackoverflow.com/questions/75649422/nlpaug-backtranslationaug-throws-runtimeerror-dataloader-worker-pids-3954-e
        if lang == "italian":
            first_model_name = 'Helsinki-NLP/opus-mt-it-en'
            second_model_name = 'Helsinki-NLP/opus-mt-en-it'
        elif lang == "english":
            first_model_name = 'Helsinki-NLP/opus-mt-en-de'
            second_model_name = 'Helsinki-NLP/opus-mt-de-en'

        first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
        first_model = MarianMTModel.from_pretrained(
            first_model_name).to(device)

        second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
        second_model = MarianMTModel.from_pretrained(
            second_model_name).to(device)

        # check the token length
        if len(first_model_tkn(text, return_tensors="pt", padding=True)['input_ids'][0]) >= 512:
            print("Text too long, skipping")
            return []

        # Translate from Original to Temporary Language
        tmp_translated_batch = perform_translation(
            text, first_model, first_model_tkn, language="en" if lang == "ita" else "de")
        # Translate Back to English
        augmented_text = perform_translation(
            tmp_translated_batch, second_model, second_model_tkn, language="it")

        # remove stopwords with nltk
        STOP_WORDS_IT = stopwords.words('italian')
        STOP_WORDS_EN = stopwords.words('english')

        STOP_WORDS = STOP_WORDS_IT + STOP_WORDS_EN

        augmented_text = " ".join(
            [word for word in augmented_text.split() if word not in STOP_WORDS])

        # MariantMT adds puntuation and special characters
        regex_punctuation_and_special = re.compile(r"[^\w\s]|_")
        augmented_text = regex_punctuation_and_special.sub(" ", augmented_text)

        augmented_text = augmented_text.lower().strip()

        augmented_texts = [augmented_text]
        return augmented_texts
    else:
        raise Exception("Method not supported")


def augument_text(df, label_to_aug, method, lang, num_aug, n_thread, bert_type="normal"):
    """
    Augments text data in a pandas DataFrame by generating new sentences based on the original text.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text data to be augmented.
        label_to_aug (str): The label of the text data to be augmented.
        method (str, optional): The method to use for text augmentation. Supported methods are "SR", "EDA", and "CWE".
        lang (str, optional): The language of the text data. Defaults to 'eng'.
        num_aug (int, optional): The number of augmented sentences to generate for each original sentence. Defaults to 2.
        n_thread (int, optional): The number of threads to use for text augmentation. Defaults to 1.

    Returns:
        pandas.DataFrame: The augmented DataFrame.
    """

    print(f"Augmenting in Language: {lang}")

    starting_time = time.time()

    for index, row in df.iterrows():
        if row['label'] == label_to_aug:
            if type(row['main_text']) == float:  # for nan values
                continue
            if len(row['main_text']) < 10:
                continue

            aug_batch = augument_sentence(
                row['main_text'][:512], lang, method, num_aug=num_aug, n_thread=n_thread, bert_type=bert_type)

            for aug_text in aug_batch:
                if index % 10 == 0:
                    print(row['main_text'][:512])
                    print(aug_text)

                # concat the new main_text and use the same label, year, subject, attachs, from, cc, filenames, size_attachs
                df = pd.concat([df, pd.DataFrame({'id': '0000000', 'year': row['year'], 'label': row['label'], 'main_text': aug_text, 'subject': row['subject'], 'attachs': row['attachs'],
                                                  'from': row['from'], 'cc': row['cc'], 'filenames': row['filenames'], 'size_attachs': row['size_attachs']}, index=[0])], ignore_index=True)

    print("Augmentation completed in ", time.time() - starting_time, " seconds")
    return df.sample(frac=1).reset_index(drop=True)


def plot_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5,
                square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_ROC_curve(y_pred, y_test, n_classes, proba=False):
    y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])

    if not proba:
        y_pred = label_binarize(y_pred, classes=[i for i in range(n_classes)])

    fig, c_ax = plt.subplots(1, 1, figsize=(12, 9))

    for (idx, c_label) in enumerate(range(n_classes)):
        fpr, tpr, thresholds = roc_curve(
            y_test[:, idx], y_pred[:, idx], drop_intermediate=True)
        # plot the roc curve for the class with the auc in percentage
        c_ax.plot(
            fpr, tpr, label=f'Class {c_label} (area= {auc(fpr, tpr)*100:0.2f}%)')

    # plot the macro average
    fpr, tpr, thresholds = roc_curve(
        y_test.ravel(), y_pred.ravel(), drop_intermediate=True)
    # dotted
    c_ax.plot(fpr, tpr, 'k--',
              label=f'Macro-average (area= {auc(fpr, tpr)*100:0.2f}%)')

    c_ax.grid(True, alpha=0.5)
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    c_ax.set_title('ROC curve')
    plt.show()


def plot_precision_recall_curve(y_preds, y_true, n_classes, proba=False):
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 9))

    y_true = label_binarize(y_true, classes=[i for i in range(n_classes)])

    if not proba:
        y_preds = label_binarize(
            y_preds, classes=[i for i in range(n_classes)])

    precision = dict()
    recall = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true[:, i], y_preds[:, i])
        # plot the precision-recall curve for the class and macro average
        c_ax.plot(recall[i], precision[i],
                  label=f'Class {i} area= ({auc(recall[i], precision[i])*100:0.2f}%)')

    # plot the macro average
    precision["macro"], recall["macro"], _ = precision_recall_curve(
        y_true.ravel(), y_preds.ravel())
    c_ax.plot(recall["macro"], precision["macro"], 'k--',
              label=f'Macro-average (area= {auc(recall["macro"], precision["macro"])*100:0.2f}%)')

    c_ax.grid(True, alpha=0.5)
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')
    c_ax.set_title('Precision-Recall curve')
    plt.show()
