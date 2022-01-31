# CMPUT 501: Intro to NLP
# Assignment 2
# Karimiab, Azamani1
# References: Stackoverflow

# Prerequisites
import re
import pandas as pd
import nltk
import sys

# Handling grammar rules that have $ in them
nltk.grammar._STANDARD_NONTERM_RE = re.compile(r'( [:\w$`\'.,/-][\w$`\'/^<>-]* ) \s*', re.VERBOSE)

# Opening a file to log errors for error analysis purposes
f = open('./error_analysis/errors.txt', 'w')


def read_train_data():
    '''
    Takes in the file path from the terminal and reads it via pandas module
    Inputs: N/A
    :return: the dataframe of the given file path
    '''
    # the second argument of the input command, from the user (data/dev)
    directory_train = sys.argv[1]
    df = pd.read_csv(directory_train, sep='\t',index_col='id')
    return df


def create_output_dataframe(df):
    '''
    Creates the output dataframe with the ground truth and prediction results
    :param df: the original dataframe
    :return: the output dataframe which contains ground truth labels from the original data and prediction column
    '''
    df_out = pd.DataFrame(columns=['ground_truth','prediction'],index=df.index)
    df_out['ground_truth']=df['label']
    return df_out


def import_grammar(grammar_path):
    '''
    Imports the grammar-related modules for parsing purposes
    :param grammar_path: path to the toy cfg grammar file
    :return: parser, an object of the ChartParser module
    '''
    grammar=nltk.data.load(grammar_path)
    parser = nltk.ChartParser(grammar)
    return parser


def print_classification_report(df_out):
    '''
    Evaluating the classification result of the grammar checking task using precision and recall
    :param df_out: the dataframe with ground truth and prediction columns
    :return: N/A
    '''
    precision, recall, TP, FN, FP, TN = calculate_precision_recall_accuracy(df_out)
    print(' '*17+'|'+' '*10+'Ground Truth')
    print(' '*17+'|'+' '*10)
    print(' '*17+'|'+' '*8+'1'+' '*13+'0')
    print('-'*50)
    print(' '*15+'1'+' '+'|'+' '*8+str(TP)+' '*5+'|'+' '*5+str(FP)+' '*6+'|')
    print('Predicted'+' '*8+'|'+'-'*32)
    print(' '*15+'0'+' '+'|'+' '*8+str(FN)+' '*5+'|'+' '*5+str(TN)+' '*6+'|')
    print(' '*17+'|'+'-'*32)

    print('precision: ', round(precision, 2))
    print('recall: ', round(recall, 2))


def calculate_precision_recall_accuracy(df_out):
    '''
    Calculating the precision and recall by using TP, FP, FN, TN 
    :param df_out: the dataframe with ground truth and prediction columns
    :return: precision, recall, TP, FN, FP, TN
    '''
    TP = FP = FN = TN = 0
    for label, pred in zip(df_out['ground_truth'], df_out['prediction']):
        # if the label is one and we also predicted one
        if label == 1 and pred == 1:
            TP += 1
        # if the label is one and we predicted zero
        elif label == 1 and pred == 0:
            FN += 1
        # if the label is zero and we predicted one
        elif label == 0 and pred == 1:
            FP += 1
        # if the label is zero and we also predicted zero
        else:
            TN += 1

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall, TP, FN, FP, TN


def write_error_to_file(txt, idx, pos):
    '''
    Saving the sentences that were incorrectly labeled by the parser to the file
    :param txt: simple template text 
    :param idx: index of the sentence
    :param pos: POS tag of the sentence
    :return: N/A
    '''
    f.write(txt)
    f.write(str(idx))
    f.write(' ')
    f.write(pos)
    f.write('\n')


def main():
    # path of the grammar file
    directory_grammar = sys.argv[2]
    # path of the output file
    directory_output = sys.argv[3]
    
    df_ipt = read_train_data()
    df_out = create_output_dataframe(df_ipt)
    parser = import_grammar(directory_grammar)

    print('Parsing has started!')
    # looping through the input dataframe
    for index, row in df_ipt.iterrows():
        # tokenizing pos tags of each dataframe's row
        tokens = row['pos'].split()
        # trying to parse the tokens using the written CFG rules
        try:
            p = list(parser.parse(tokens))
            # if len(p) = 0, it means that the sentence contains grammatical errors - assign the prediction to 1
            if len(p) == 0:
                df_out.loc[index, 'prediction'] = 1
                if row['label'] == 0:
                    write_error_to_file('real label: 0, predicted label: 1 ', index, row['pos'])
            # if len(p) != 0, it means the grammar is correct - assign the prediction to 0
            else:
                df_out.loc[index, 'prediction'] = 0
                if row['label'] == 1:
                    write_error_to_file('real label: 1, predicted label: 0 ', index, row['pos'])
        # in cases where we have not covered a specific tag
        except ValueError as e:
            df_out.loc[index,'prediction']=1
            if row['label'] == 0:
                write_error_to_file('real label: 0, predicted label: 1 ', index, row['pos'])

    print('Parsing has ended!')
    
    # Save df_out to output.tsv
    df_out.to_csv(directory_output, sep='\t')
    print_classification_report(df_out)


if __name__ == '__main__':
    main()
