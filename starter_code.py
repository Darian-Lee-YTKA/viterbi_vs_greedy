import os
import nltk
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import nltk
nltk.download('averaged_perceptron_tagger')
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def evaluate(test_sentences, tagged_test_sentences, save=False):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    accuracy = accuracy_score(gold, pred)
    cm = confusion_matrix(gold, pred)
    tags = sorted(set(gold).union(set(pred)))
    if save:
        report = metrics.classification_report(gold, pred)
        output_path = "/Users/darianlee/PycharmProjects/201_hw3/results"
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Evaluation report saved to {output_path}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average='macro', zero_division=0
    )


    plt.figure(figsize=(18, 18))
    sns.heatmap(cm, annot=False, cmap='Reds', xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Accuracy: {accuracy}")


    cm_df = pd.DataFrame(cm, index=tags, columns=tags)


    cm_no_diag = cm_df.where(np.eye(cm_df.shape[0], dtype=bool) == False)


    misclassified_pairs = cm_no_diag.stack().sort_values(ascending=False)

    print("Most commonly confused tag pairs:")
    print(misclassified_pairs.head())

    return precision, recall, f1, accuracy


def get_token_tag_tuples(sent):
    return([nltk.tag.str2tuple(t) for t in sent.split()])

def get_tagged_sentences(text):
    sentences = []

    blocks = text.split("======================================")
    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "")
            if sent != "":
                sentences.append(sent)
    return sentences

def load_treebank_splits(datadir):

    train = []
    dev = []
    test = []

    print("Loading treebank data...")

    for subdir, dirs, files in os.walk(datadir):

       for filename in files:
           if filename.endswith(".pos"):
               filepath = subdir + os.sep + filename
               with open(filepath, "r") as fh:
                   text = fh.read()
                   if int(subdir.split(os.sep)[-1]) in range(0, 19):
                       train += get_tagged_sentences(text)

                   if int(subdir.split(os.sep)[-1]) in range(19, 22):
                       dev += get_tagged_sentences(text)

                   if int(subdir.split(os.sep)[-1]) in range(22, 25):
                       test += get_tagged_sentences(text)

    print("Train set size: ", len(train))
    print("Dev set size: ", len(dev))
    print("Test set size: ", len(test))

    return train, dev, test
def Darians_amazing_hmm(train, alpha):
    #train will be in the form [[(sentence1_token1, tag1), (sentence1_token2, sentence1_tag2)], [sentence2...]]
    transition_probabilities = {} # will hold [tag1: {tag2: count, tag3: count}} representing the count of tokens that start in tag1 and go to tag2
    ground_probabilities = {} # will hold count(tag1). important cause p(tag1->tag2) = count(tag1 -> tag2)/count(tag1)

    #########      transition probs    #############
    # getting counts
    for sentence in train:
        sentence = [("<S>", "<S>")] + sentence + [("<E>", "<E>")]
        for tuple_index in range(len(sentence) - 1):

            tag1 = str(sentence[tuple_index][1])
            tag2 = str(sentence[tuple_index + 1][1])
            if tag1 in transition_probabilities:

                if tag2 in transition_probabilities[tag1]:
                    transition_probabilities[tag1][tag2] += 1 #add 1 to count of tag1->tag2
                else:
                    transition_probabilities[tag1][tag2] = 1
            else:

                transition_probabilities[tag1] = {tag2: 1}

            if sentence[tuple_index][1] in ground_probabilities:
                ground_probabilities[sentence[tuple_index][1]] += 1
            else:
                ground_probabilities[sentence[tuple_index][1]] = 1 # will keep track of the denominator. Example p(NN -> NNS) should be count(NN -> NNS)/ count(NN)
    for key in list(transition_probabilities)[:5]:
        print(key, transition_probabilities[key])

    # converting counts to probs
    for from_tag in transition_probabilities:

        for to_tag in transition_probabilities[from_tag]:
            if from_tag in ground_probabilities:
                #transition_probabilities[from_tag][to_tag] = (transition_probabilities[from_tag].get(to_tag,0)) / (ground_probabilities[from_tag])  # smoothing with alpha = 1
                transition_probabilities[from_tag][to_tag] = (transition_probabilities[from_tag].get(to_tag, 0) + alpha) / (ground_probabilities[from_tag] + alpha * len(transition_probabilities)) # smoothing with alpha = 1
            else:
                print("this shouldnt happen")
                print(from_tag)
    for key in list(transition_probabilities)[:1]:
        print(key, transition_probabilities[key])

    ############# emission probabilities ############
    emission_probabilities = {}
    tag_counts = {} #this should equal ground probabilities if I am doing everything right

    for sentence in train:
        sentence = [("<S>", "<S>")] + sentence + [("<E>", "<E>")]
        for tuple in sentence:
            tag = str(tuple[1])
            emission = str(tuple[0])
            if tag in emission_probabilities:
                if emission in emission_probabilities[tag]:
                    emission_probabilities[tag][emission] +=1
                else:
                    emission_probabilities[tag][emission] = 1
            else:
                emission_probabilities[tag] = {emission:1}

            if tuple[1] in tag_counts:
                tag_counts[tuple[1]] += 1
            else:
                tag_counts[tuple[1]] = 1

    print("tag_counts == ground_probabilities: ", tag_counts == ground_probabilities)
    for key in list(emission_probabilities)[:2]:
        print(key, emission_probabilities[key])
    for key in tag_counts:
        if key not in ground_probabilities:
            print(f"key {key} isnt in ground_probabilities") # the only difference is one contains the <e> token and one doesnt, which is expected
        elif tag_counts[key] != ground_probabilities[key]:
            print(f"value of key {key} is different:")
            print(f"  tag_counts: {tag_counts[key]}")
            print(f"  ground_probabilities: {ground_probabilities[key]}")


    for key in ground_probabilities:
        if key not in tag_counts:
            print(f"key {key}is not in tag_counts")

    # converting counts to probs
    for tag in emission_probabilities:
        for emission in emission_probabilities[tag]:

            emission_probabilities[tag][emission] = (emission_probabilities[tag].get(emission, 0) + alpha) / (tag_counts[tag] + alpha * len(emission_probabilities))
            #emission_probabilities[tag][emission] = (emission_probabilities[tag].get(emission, 0)) / (tag_counts[tag])

    for key in list(emission_probabilities)[:2]:
        print(key, emission_probabilities[key])




    return transition_probabilities, emission_probabilities, list(tag_counts.keys())

import pandas as pd
import numpy as np

class Beautiful_viterbi_table:
    def __init__(self, tags, seq_len):
        self.table = pd.DataFrame(np.zeros((seq_len, len(tags))), columns=tags)
        print("🟪🟪🟪 PRINTING SELF>TABLE 🟪🟪🟪")
        print(self.table)
        # df where the rows will be the prob of each tag at each timestamp
        # initialize to have the same number of rows as seq_len with all 0s

    def change_probs_at_row(self, row_idx, row_of_probs):
        if len(row_of_probs) != len(self.table.columns):
            print("the row passed in is a different size than the number of tags")
            return

        self.table.loc[row_idx] = row_of_probs


# for getting the greedy
def greedy(transition_probabilities, emission_probabilities, token_sequence, tags):
    token_sequence = ["<S>"] + token_sequence + ["<E>"]
    path = []

    prev_tag = "<S>"

    for t in range(1, len(token_sequence) - 1):
        current_token = token_sequence[t]
        max_prob = -np.inf
        best_tag = None


        for curr_tag in tags:
            transition_log = np.log(transition_probabilities.get(prev_tag, {}).get(curr_tag, 1e-10))
            emission_log = np.log(emission_probabilities.get(curr_tag, {}).get(current_token, 1e-10))

            log_prob = transition_log + emission_log
            if log_prob > max_prob:
                max_prob = log_prob
                best_tag = curr_tag


        path.append((current_token, best_tag))
        prev_tag = best_tag

    path.append(("<E>", "<E>"))

    print("🧡🧡🧡 printing path: ", path)
    return path

# for getting the baseline based only on emission prob
def baseline(transition_probabilities, emission_probabilities, token_sequence, tags):
    token_sequence = ["<S>"] + token_sequence + ["<E>"]
    path = []

    prev_tag = "<S>"

    for t in range(1, len(token_sequence) - 1):
        current_token = token_sequence[t]
        max_prob = -np.inf
        best_tag = None


        for curr_tag in tags:
            transition_log = np.log(transition_probabilities.get(prev_tag, {}).get(curr_tag, 1e-10))
            emission_log = np.log(emission_probabilities.get(curr_tag, {}).get(current_token, 1e-10))

            log_prob = emission_log
            if log_prob > max_prob:
                max_prob = log_prob
                best_tag = curr_tag


        path.append((current_token, best_tag))
        prev_tag = best_tag

    path.append(("<E>", "<E>"))

    print("🧡🧡🧡 printing path: ", path)
    return path

def viterbi(transition_probabilities, emission_probabilities, token_sequence, tags):

    token_sequence = ["<S>"] + token_sequence + ["<E>"]
    table = Beautiful_viterbi_table(tags, len(token_sequence))
    backtrace = [{} for _ in range(len(token_sequence))]
    print("🟪🟪🟪 these are the tags we got: ", tags)
    # make 0 for <S> and -inf for the rest in the first row (log space)
    row1 = []
    for tag in tags:
        if tag == "<S>":
            row1.append(0)  # log(1) = 0
        else:
            row1.append(-np.inf)  # log(0) = -inf
    table.change_probs_at_row(0, row1)

    # fill in the rest of the table
    for t in range(1, len(token_sequence)):
        current_token = token_sequence[t]
        prev_row = table.table.iloc[t - 1]  # prev row log-probabilities
        current_row = []

        for curr_tag in tags:
            max_log_prob = -np.inf
            best_prev_tag = None
            for prev_tag in tags:
                # log transition prob
                transition_log = np.log(transition_probabilities.get(prev_tag, {}).get(curr_tag, 1e-10))
                # log emission prob
                emission_log = np.log(emission_probabilities.get(curr_tag, {}).get(current_token, 1e-10))

                log_prob = prev_row[prev_tag] + transition_log + emission_log
                ## note: this was all working fine and dandy when I ran it on the test set.
                # For some reason it is getting weird errors for dev
                # I think is is because the sentences in dev are less similar to the train examples
                #print("log_prob: ", log_prob)
                '''if isinstance(log_prob, pd.Series):
                    print("🟥 log_prob is a pandas Series")
                    print("prev_row[prev_tag]: ", prev_row[prev_tag])
                    print("transition_log : ", transition_log)
                    print("emission_log :", emission_log)
                    print("log prob: ", log_prob)
                    print("token_sequence: ", token_sequence)
                else:
                    print("🟢log_prob is not a pandas Series")
                    print("prev_row[prev_tag]: ", prev_row[prev_tag])
                    print("transition_log : ", transition_log)
                    print("emission_log :", emission_log)
                    print("log prob: ", log_prob)
                    print("token_sequence: ", token_sequence)'''


                # NOTE: this shouldnt happen anymore. I was doing
                """if isinstance(log_prob, pd.Series) or np.isnan(log_prob) or log_prob == -np.inf:
                    log_prob = np.random.uniform(-100, -80)
                    #print("Random log_prob chosen for token:", current_token, "with tag:", curr_tag)"""

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_prev_tag = prev_tag
            current_row.append(max_log_prob)
            backtrace[t][curr_tag] = best_prev_tag

        # adding the current row to the table
        table.change_probs_at_row(t, current_row)

    # backtracing to find the most probable sequence
    most_probable_path = []
    last_row = table.table.iloc[-1]
    current_tag = last_row.idxmax()  # get tag with the highest log-probability in the last row
    best_prob = last_row.max()
    most_probable_path.append(current_tag)

    for t in range(len(token_sequence) - 1, 0, -1):  # to skip <E> and <S>
        current_tag = backtrace[t][current_tag]
        most_probable_path.append(current_tag)

    most_probable_path.reverse()  # reverse to get correct order

    print(table.table)
    print(most_probable_path)
    return (most_probable_path, token_sequence, best_prob)







def main():
    # uncomment  this part to experiment with different alpha values (spoiler 1 is best)
    # run_experiments()

    # uncomment this part to get the greedy results on the test set
    #run_greedy()

    # uncomment this part to get the baseline results on the test set
    #run_baseline()

    # uncomment this part to get the viterbi results for test (all predictions saved as a json file)
    # warning, this function takes a very long time to run
    #run_test()

    # uncomment to load the json file and evaluate it in comparision to the test data
    #evaluate_viterbi_tes()



def run_experiments():
    datadir = "/Users/darianlee/PycharmProjects/201_hw3/data/penn-treeban3-wsj/wsj"
    train, dev, test = load_treebank_splits(datadir)
    best_a = None
    best_f1 = -1
    stats = {}
    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    dev_sentences = [get_token_tag_tuples(sent) for sent in dev]
    dev_sentences = dev_sentences[:100]
    print("a favor of train 🍭😋")
    print(train_sentences[:2])
    print("a flavor of dev 🍭😋")
    print(dev_sentences[:2])
    untagged_dev = []
    for sentence in dev_sentences:
        new_sentence = []
        for token_tag_tuple in sentence:
            new_sentence.append(token_tag_tuple[0])
        untagged_dev.append(new_sentence)

    print("untagged dev!!!!! : ", untagged_dev[:5])

    for alpha in range(1,6):


        tagged_dev_sentences = []

        transition_probabilities, emission_probabilities, tags = Darians_amazing_hmm(train_sentences, alpha)
        print(tags)

        for sentence in untagged_dev:
            print(sentence)
            best_path = viterbi(transition_probabilities, emission_probabilities, sentence, tags)
            tags_v = best_path[0]
            words = best_path[1]
            tags_v = tags_v[1:-1]
            words = words[1:-1]

            tagged_dev_sentences.append([(word, tag) for word, tag in zip(words, tags_v)])

        p, r, f1, acc = evaluate(dev_sentences, tagged_dev_sentences)
        print(f"For alpha {alpha} we got p, r, f1 = {p:.2f}, {r:.2f}, {f1:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_a = alpha
        key = "alpha " + str(alpha)
        stats[key] = {"percision": p, "recall:": r, "f1": f1, "acc": acc}
    print("BEST ALPHA WAS: ", best_a, "WITH F! SCORE: ", best_f1)
    print("all stats: ", stats)

def run_greedy():
    datadir = "/Users/darianlee/PycharmProjects/201_hw3/data/penn-treeban3-wsj/wsj"
    train, dev, test = load_treebank_splits(datadir)
    alpha = 1
    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    untagged_test = []
    for sentence in test_sentences:
        new_sentence = []
        for token_tag_tuple in sentence:
            new_sentence.append(token_tag_tuple[0])
        untagged_test.append(new_sentence)





    tagged_test_sentences = []

    transition_probabilities, emission_probabilities, tags = Darians_amazing_hmm(train_sentences, alpha)
    print(tags)

    for sentence in untagged_test:
            print(sentence)
            best_path = greedy(transition_probabilities, emission_probabilities, sentence, tags)
            best_path = best_path[:-1]
            tagged_test_sentences.append(best_path)




    p, r, f1, acc = evaluate(test_sentences, tagged_test_sentences)
    print("🤑🤑 Results for greedy model:")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1-Score: {f1}")
    print(f"Accuracy: {acc}")

def run_baseline():
    datadir = "/Users/darianlee/PycharmProjects/201_hw3/data/penn-treeban3-wsj/wsj"
    train, dev, test = load_treebank_splits(datadir)
    alpha = 1
    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    untagged_test = []
    for sentence in test_sentences:
        new_sentence = []
        for token_tag_tuple in sentence:
            new_sentence.append(token_tag_tuple[0])
        untagged_test.append(new_sentence)





    tagged_test_sentences = []

    transition_probabilities, emission_probabilities, tags = Darians_amazing_hmm(train_sentences, alpha)
    print(tags)

    for sentence in untagged_test:
            print(sentence)
            best_path = baseline(transition_probabilities, emission_probabilities, sentence, tags)
            best_path = best_path[:-1]
            tagged_test_sentences.append(best_path)




    p, r, f1, acc = evaluate(test_sentences, tagged_test_sentences)
    print("⚾️⚾️ Results for baseline model:")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1-Score: {f1}")
    print(f"Accuracy: {acc}")
def run_test():
    # Set path for datadir
    datadir = "/Users/darianlee/PycharmProjects/201_hw3/data/penn-treeban3-wsj/wsj"

    train, dev, test = load_treebank_splits(datadir)

    ## For evaluation against the default NLTK POS tagger

    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    train_sentences = [get_token_tag_tuples(sent) for sent in train]

    print("a favor of train 🍭😋")
    print(train_sentences[:5])
    print("a flavor of test 🍭😋")
    print(test_sentences[:5])

    tagged_test_sentences = []
    print(test_sentences[:5])
    transition_probabilities, emission_probabilities, tags = Darians_amazing_hmm(train_sentences, 0)
    untagged_test = []
    for sentence in test_sentences:
        new_sentence = []
        for token_tag_tuple in sentence:
            new_sentence.append(token_tag_tuple[0])
        untagged_test.append(new_sentence)

    print("🟪🟪", untagged_test[:2])
    best_paths = []
    print(len(untagged_test))
    print(len(train_sentences))


    for sentence in untagged_test:
        best_path = viterbi(transition_probabilities, emission_probabilities, sentence, tags)
        best_paths.append(best_path)
    import json

    # change the directory to your desired directory
    with open("/Users/darianlee/PycharmProjects/201_hw3/best_paths2.json", "w") as f:
        json.dump(best_paths, f)

def evaluate_viterbi_test():
    import json
    datadir = "/Users/darianlee/PycharmProjects/201_hw3/data/penn-treeban3-wsj/wsj"

    train, dev, test = load_treebank_splits(datadir)

    ## For evaluation against the default NLTK POS tagger

    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    train_sentences = [get_token_tag_tuples(sent) for sent in train]

    print("a favor of train 🍭😋")
    print(train_sentences[:5])
    print("a flavor of test 🍭😋")
    print(test_sentences[:5])

    tagged_test_sentences = []
    print(test_sentences[:5])


    untagged_test = []
    for sentence in test_sentences:
        new_sentence = []
        for token_tag_tuple in sentence:
            new_sentence.append(token_tag_tuple[0])
        untagged_test.append(new_sentence)

    print("🟪🟪", untagged_test[:2])

    print(len(untagged_test))
    print(len(train_sentences))

    with open('/Users/darianlee/PycharmProjects/201_hw3/best_paths.json', 'r') as f:
        data = json.load(f)

    for entry in data:
        tags = entry[0]
        words = entry[1]
        # to remove <S> and <E>
        tags = tags[1:-1]
        words = words[1:-1]

        tagged_test_sentences.append([(word, tag) for word, tag in zip(words, tags)])

    print(tagged_test_sentences[:5])

    print(len(tagged_test_sentences))
    print(len(test_sentences))
    evaluate(test_sentences, tagged_test_sentences, True)


if __name__ == "__main__":
    main()


