# @ Hamad Alrashid
# Script for benchmarking different tokenizers in BM25 in pyserini
# Miracl, Mrtydi datasets
# Supports mutliprocessing for performing the SQL search


from tqdm import tqdm 
import os, json, yaml, time, re, datasets, glob, sys
from argparse import ArgumentParser
from datasets import load_dataset
import sentencepiece as spm
from datetime import datetime
from transformers import AutoTokenizer
sys.path.append('/home/hamad/benchmark/pyserini/')
from pyserini.analysis import Analyzer, get_lucene_analyzer




languages = [
    ['ar', 'arabic'],
    ['bn', 'bengali'],
    ['en', 'english'],
    ['es', 'spanish'],
    ['fa', 'persian'],
    ['fi', 'finnish'],
    ['fr', 'french'],
    ['hi', 'hindi'],
    ['id', 'indonesian'],
    ['ja', 'japanese'],
    ['ko', 'korean'],
    ['ru', 'russian'],
    ['sw', 'swahili'],
    ['te', 'telugu'],
    ['th', 'thai'],
    ['zh', 'chinese'],
    ['de', 'german'],
    ['yo', 'yoruba']
]

def get_lang(language):

    '''
    returns either the id or the name of the given language

    '''
    for l in languages:
        if language.lower() in l:
            
            # Return name given ID
            if len(language) == 2:
                return l[1]
            else:
                # Return ID given name
                return l[0]
            
        
    return None

def get_time():
    '''
    Print current time in HMS format
    '''
    print("[Time: {time}]".format(time = datetime.now().strftime("%H:%M:%S")))




def generate_dataset(path, language, split, num_proc, use_auth_token=True):

    '''
    Download a huggingface dataset using load_dataset and return the dataset['split']
    '''
    print(f"[Downloading {path}, {language}, {split}" )

    # for miracl, a num_proc > 1 breaks the function for some reason
    
    if 'miracl' in path:
        num_proc = None
    
    
    data = load_dataset(path, language, use_auth_token=use_auth_token)#, num_proc=num_proc)

    return data
    


def tokenize_dataset(config_file, dataset, language, tokenizer_path):

    '''
    Tokenize the corpus & queries of dataset using the spm tokenizer located at tokenizer_path.
    Return an array that contains the path of the queries and corpus
    '''

    start = time.time()
    

    sp = spm.SentencePieceProcessor(tokenizer_path) 
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)    
    #tokenizer = Analyzer(get_lucene_analyzer(language='es'))
    # Returns the tokenized text as a string 

    def tokenize(text):
        
        return " ".join(sp.EncodeAsPieces(text)) # Sentencepiece tokenizer
        #return ' '.join(tokenizer.tokenize(text)) # Huggingface tokenizer
        #return ' '.join(tokenizer.analyze(text)) # Lucene analyzer
    
    
    # Creating the directories for generating the dataset
    dataset_dir = os.path.join("datasets", f"{dataset}-{language}-tokenized") # Where to save the dataset
    queries_path = os.path.join(dataset_dir, "queries")
    corpus_path = os.path.join(dataset_dir, "corpus")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(queries_path, exist_ok=True)
    os.makedirs(corpus_path, exist_ok=True)


    # Loading the yaml config file
    f = open(config_file, "r") 
    config = yaml.safe_load(f)
    f.close()


    num_proc = config['NUM_PROC']

    
    # Processing the queries
    def tokenize_queries(entry):
        entry['query'] = tokenize(entry['query'])
        return entry
    

    path = config['datasets'][dataset]['queries_path']
    dataset_language = language if language in config['datasets'][dataset]['languages'] else get_lang(language)
    split = config['datasets'][dataset]['queries_split']
    
    dataset_generated = generate_dataset(path=path, language=dataset_language, split=split,  num_proc=num_proc)
    queries = dataset_generated[split].map(tokenize_queries,  num_proc=num_proc, load_from_cache_file=False)

    queries_output_path = os.path.join(queries_path, f"{split}.jsonl")
    queries.to_json(queries_output_path, batch_size=(datasets.config.DEFAULT_MAX_BATCH_SIZE)*10, num_proc=num_proc, force_ascii=False)
    #dataset_generated.cleanup_cache_files()
    
    #dataset_language = language if language in config['datasets'][dataset]['languages'] else get_lang(language)
    print(f"[Queries saved at {queries_output_path}]")


    # Processing the corpus
    def tokenize_corpus(entry):
        entry['title'] = tokenize(entry['title'])
        entry['text'] = tokenize(entry['text'])
        return entry
    
    path = config['datasets'][dataset]['corpus_path']
    split = config['datasets'][dataset]['corpus_split']

    dataset_generated = generate_dataset(path=path, language=dataset_language, split=split, num_proc=num_proc)
    corpus = dataset_generated[split].map(tokenize_corpus,  num_proc=num_proc, load_from_cache_file=False)


    num_shards = num_proc
    for shard_idx in range(num_shards):
        shard = corpus.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        corpus_output_path = os.path.join(corpus_path, f"{split}-{shard_idx}.jsonl")
        shard.to_json(corpus_output_path, batch_size=(datasets.config.DEFAULT_MAX_BATCH_SIZE)*10, num_proc=num_proc, force_ascii=False)
    
    dataset_generated.cleanup_cache_files()
    end = time.time()
    print(f"[Done Tokenizing  {dataset}-{language} with {tokenizer_path} in {end-start} s]")
    get_time()

    return [corpus_path, queries_output_path]



def index_corpus(corpus_path, threads, index_dir):

    '''
    Index the corpus using pyserini indexer
    '''

    print(f"Indexing {corpus_path}")
    get_time()

    os.makedirs(index_dir, exist_ok=True)
    cmd = f'python -m pyserini.index.lucene \
        -collection MrTyDiCollection \
        -generator DefaultLuceneDocumentGenerator \
        -threads {threads} \
        -input {corpus_path} \
        -index {index_dir} \
        -storePositions \
        -storeRaw \
        -storeDocvectors \
        -pretokenized \
        -optimize'
    
    os.system(cmd)

    print("Done Indexing")
    get_time()

    return index_dir



def convert_queries_to_tsv(input):
    '''
    Convert input file (jsonl queries) to tsv format for performing a batch search
    '''
    
    output = input.replace("jsonl", "tsv")
    with open(input, "r") as jsonl_file:
        with open(output, "w") as tsv_file:
            json_data = list(jsonl_file) # list contains all queries as json

            print(f"Processing {len(json_data)} queries...")
            for j in tqdm(json_data):
                fields = json.loads(j)
                line = f"{fields['query_id'].strip()}\t{fields['query'].strip()}\n"
                tsv_file.write(line)
    print(f"Done converting {input} to tsv format")
    return output

    
            

def batch_search(dataset, index_path, topics_path, output_path, num_threads=32):

    '''
    Perform a batch search
    '''
    
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "results.txt")


    
    if dataset == 'mrtydi': 
        cmd = f"python -m pyserini.search.lucene --threads 16 --batch-size 128 \
                --pretokenized \
                --topics {topics_path} \
                --index {index_path} \
                --output {output_path} \
                --bm25 \
                --hits 100"
    else: 
        cmd = f"python -m pyserini.search.lucene \
        --pretokenized \
        --topics {topics_path} \
        --index {index_path} \
        --output {output_path} \
        --batch 128 \
        --threads 16 \
        --bm25 \
        --k1 1.2\
        --b 0.75 \
        --hits 100"
    
    os.system(cmd)

    return output_path
    

def evaluate(dataset, qrels, search_results, output):


    if dataset == 'mrtydi':
        cmd1 = f"python -m pyserini.eval.trec_eval \
             -c -M 100 -m recip_rank {qrels} {search_results} > {output}"
        
        cmd2 = f"python -m pyserini.eval.trec_eval \
                -c -m recall.100 {qrels} \
                {search_results} >> {output}"
        
    else:
        cmd1 = f"python -m pyserini.eval.trec_eval \
                -c -M 100 -m ndcg_cut.10 {qrels} \
                {search_results} > {output}"
        cmd2 = f"python -m pyserini.eval.trec_eval \
                -c -m recall.100 {qrels} \
                {search_results} >> {output}"
        
    os.system(cmd1)
    os.system(cmd2)

    '''
    # retrieve mrr and recall scores and return them
    result = {}

    with open(output, "r") as reader:
        lines = reader.readlines()
        for i in range(0, len(lines)):
            if re.match(r"recip", lines[i]):
                break
        mrr = re.search(r"all\s(.+)$", lines[i]).group(1)
        recall = re.search(r"all\s(.+)$", lines[i+1]).group(1)
        result['MRR'] = mrr
        result['RECALL'] = recall

    print(f"Results saved at {output}")
    return result
    '''
    return output

def clean(y):
    ''' 
    Removes datasets, indexes, runs
    '''
    print("Cleaning...")
    
    os.system(f"rm -rf {y['RUNS_PATH']}")
    os.system(f"rm -rf {y['INDEX_PATH']}")
    os.system(f"rm -rf datasets")

    

def benchmark(tokenizer_path, config_file, dataset, language):

    ''' 
    Benchmark the tokenizer and returns a dict in the following format: {'MRR': v1, 'RECALL': v2}
    '''
    start = time.time()
    
    config_name = f"{tokenizer_path.replace('/', '-')}-{dataset}-{language}"

    with open(config_file, "r") as f:
        y = yaml.safe_load(f)

        # Tokenizing the dataset
        corpus_path, queries_path = tokenize_dataset(config_file, dataset, language, tokenizer_path)
        

        # Indexing
        index_path = os.path.join(y['INDEX_PATH'], dataset, config_name)
        index_dir = index_corpus(corpus_path, y['NUM_PROC'], index_path)


        # Search
        runs_path = os.path.join(y['RUNS_PATH'], config_name)
        queries_path = convert_queries_to_tsv(queries_path)
        search_results = batch_search(dataset, index_dir, queries_path, runs_path)    


        # Retrieve results
        qrels = y['datasets'][dataset]['qrels_path'] 
        qrels = qrels.replace('LANGID', language) if len(language) == 2 else qrels.replace('LANGID', get_lang(language))
        os.makedirs(y['EVAL_PATH'], exist_ok=True)

        eval_file = os.path.join(y['EVAL_PATH'], f'{config_name}.txt')
        results = evaluate(dataset, qrels, search_results, eval_file)


        # Delete datasets and indexes 
        clean(y)

        end = time.time()
        print(f"[Done in {(end-start) / 60} minutes]")

        return results


def main():
    
    # tokenizers to benchmark
    tokenizers = ['ec2.model'] #['bert-base-multilingual-uncased', 'xlm-roberta-large'] #["spm1-uncased.model", "merged.model" ]

    config_file = "parameters.yaml"

    mrtydi_languages = ['arabic', 'bengali', 'english', 'indonesian','finnish', 'korean', 'russian', 'swahili', 'telugu', 'thai', 'japanese']
    miracl_languages = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh', 'de', 'yo']

    datasets = {'miracl': miracl_languages}

    log = {}

    start = time.time()
    done = 0

    for dataset in datasets:

        languages = datasets[dataset]

        for language in tqdm(languages):
            
            for tokenizer_path in tokenizers:

                # Skip language if tokenizer has already been tested
                if os.path.exists(f"eval/{tokenizer_path}-{dataset}-{language}.txt"):
                    continue
                fails = 0

                while True:
                    if fails > 2:
                        break
                    try:
                        benchmark(tokenizer_path, config_file, dataset, language)
                        done += 1
                        break  
                    except Exception as e:
                        print(e)
                        
                        log[f'{tokenizer_path}-{dataset}-{language}'] = e
                        fails += 1
                        
            #os.system("rm -rf /home/hamad/.cache/huggingface/datasets")
            #os.system("mkdir /home/hamad/.cache/huggingface/datasets")

    print(log)
    print(done)
    print(f'Done in {((time.time()) - start) / 60 } minutes')

if __name__ == '__main__':
    main()