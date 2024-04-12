# @ Hamad Alrashid
# Script for benchmarking FTS5 
# Pretokenizing queries & corpus using Sentencepiece vs Default tokenizer
# Miracl Dataset (16 lanugages)
# Supports mutliprocessing for performing the Lucene search



import time, sqlite3, datasets, tqdm, re, os
import sentencepiece as spm
from joblib import Parallel, delayed




num_proc = 32 # For multiprocessing

# Initializing the Sentencepiece tokenizer
tokenizer_path='/home/hamad/benchmark/fts5/ftsqlitesp/ec2.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)


def tokenize(text):
    '''
    Tokenize the string using Sentencepiece and joins the tokens with a space
    '''

    return " ".join(sp.EncodeAsPieces(text)) 


def tokenize_corpus(entry):
    '''
    Tokenize a Miracl doc
    '''
    entry['title'] = tokenize(entry['title'])
    entry['text'] = tokenize(entry['text'])

    return entry

def tokenize_queries(entry):
    '''
    Tokenize a Miracl query
    '''
    entry['query'] = tokenize(entry['query'])
    return entry


def batch_iterable(iterable, n=100):
    '''
    For multiprocessing: returns a batch of size n for each process
    '''
    batch = []
    for line in iterable:
        batch.append(line)
        if len(batch) == n:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
        batch = []
    return

def load_into_db(language, tokenizer):

    '''
    Load the Miracl dataset into an sqlite db.
    If tokenizer is sp, the dataset will be pretokenized using sentencepiece and then inserted into the db.
    Returns the name of the database
    '''

    print(f"Loading {language} corpus")
    corpus_dataset = datasets.load_dataset('miracl/miracl-corpus', f'{language}')
    documents = corpus_dataset['train']

    if tokenizer == 'sp':
        print(f'Tokenizing {language} corpus')
        documents = corpus_dataset['train'].map(tokenize_corpus,  num_proc=32, load_from_cache_file=False)


    conn = sqlite3.connect(f'{language}-{tokenizer}.db')
    c = conn.cursor()

    # For loading an FTS5 extension

    '''
    conn.enable_load_extension(True)
    c.execute("SELECT load_extension('~/Signal-FTS5-Extension/target/debug/libsignal_tokenizer.so', 'sqlite3_libsignal_tokenizer_init')")
    exit()
    '''


    c.execute("DROP TABLE IF EXISTS documents")
    conn.commit()

    c.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(docid, title, text, tokenize = "trigram");
    ''')
            
    print('Inserting docs')
    for item in tqdm.tqdm(documents):
        c.execute("INSERT INTO documents(docid, title, text) VALUES (?, ?, ?)", 
                (item['docid'], item['title'], item['text']))

    conn.commit()
    conn.close()
    corpus_dataset.cleanup_cache_files()

    return f'{language}-{tokenizer}.db'


def search_fts5(language, tokenizer, db_name):
    '''
    Run all Miracl queries. 
    If tokenizer is sp, the queries will be pretokenized using Sentencepiece
    Returns the path of the trec formatted runfile that contains the search results and the average time for running all queries
    '''

    runs_out = f'{language}-{tokenizer}-runs.txt'


    def clean_text(text):
        '''
        Util func for cleaning the query, removing ' and " symbols
        '''
        # Replace any character that is not a space, digit, upper or lower case letter, or underscore with a space
        #text = re.sub(r'[^\s\w]', ' ', text).lower()   
        text = re.sub(r'[\'|\"]+', ' ', text).strip()
        # Collapse consecutive spaces
        #text = re.sub(r'\s+', ' ', text).strip()
    

        return text
    
    def batch_process(queries):
        '''
        Proccess the queries
        '''
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        table_name = 'documents'
        title_weight = 1
        text_weight = 1
        bm25_txt = f'bm25({table_name})' #f'bm25({table_name}, 0, {title_weight}, {text_weight})' # multi field search
        k = 100 # LIMIT number of results
        results = []
        latency = 0 
        for entry in tqdm.tqdm(queries):
            
            
            # Process the query
            query = clean_text(entry['query'])
            #query = query.replace(' ', ' OR ')
            words = query.split(' ')
            query = ' OR '.join(["\"" + word + "\"" for word in words])

            q_txt = f"SELECT docid, -1 * {bm25_txt} FROM {table_name} WHERE {table_name} MATCH '({query})' ORDER BY {bm25_txt} LIMIT {k}"
            
            # Search
            start = time.time()
            c.execute(q_txt)
            end = time.time()

            latency += end-start
            documents_results = c.fetchall()
            
            # TREC Formating 
            for rank, result in enumerate(documents_results):
                docid, score = result
                line = f'{entry["query_id"]} Q0 {docid} {rank+1} {score} X\n'
                results.append(line)
        print(f'Processed {len(queries)} queries')

        return results, latency
    
    print(f"Loading {language} queries")
    queries_dataset = datasets.load_dataset('miracl/miracl', language)['dev']
    queries = queries_dataset

    if tokenizer == 'sp':
        print('Tokenizing queries')
        queries_tokenized = queries.map(tokenize_queries,  num_proc=32, load_from_cache_file=False)
        queries = queries_tokenized


    print(f"Spawning {num_proc} processes")
    pool = Parallel(n_jobs=num_proc, verbose=10)
    average_latency = 0
    print('Searching')

    for batch, latency  in pool([delayed(batch_process)(batch) for batch in batch_iterable(queries, (len(queries)//num_proc)+1)]):
        with open(runs_out, 'a') as fout:
            print(".", end='')
            fout.writelines(batch)
        average_latency += latency

    queries.cleanup_cache_files()

    average_latency = average_latency / len(queries) # Averaging the latency


    return runs_out, average_latency


def evaluate(run_file, qrels_file, average_latency):
    '''
    Evaluate the run file with pyserini.eval.trec_eval
    Returns the path of the results file
    '''
    eval_out = f'eval/eval-{run_file}'
    cmd1 = f"python -m pyserini.eval.trec_eval    \
                    -c -M 100 -m ndcg_cut.10 {qrels_file} \
                    {run_file} > {eval_out}"
    cmd2 = f"python -m pyserini.eval.trec_eval \
            -c -m recall.100 -m recip_rank {qrels_file} \
            {run_file} >> {eval_out}"

    os.system(cmd1)
    os.system(cmd2)
    os.system(f'echo \"Average_latency: {average_latency}\" >> {eval_out}')
    return eval_out


def clean():
    '''
    Deletes the sql database, Miracl datasets folder.
    '''

    datasets_path = "/home/hamad/.cache/huggingface/datasets"
    print("Cleaning...")
    os.system('rm -rf *.db')
    #os.system(f'rm -rf {datasets_path}')
    #os.system(f'mkdir {datasets_path}')

def main():
    
    miracl_languages = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh', 'de', 'yo']
    tokenizers = ['default', 'sp']

    # Tests = 36
    for language in tqdm.tqdm(miracl_languages):

        for tokenizer in tokenizers:
            print(f"Benchmarking {language}-{tokenizer}")

            # Skip language if tokenizer has already been tested
            if os.path.exists(f"eval/eval-{language}-{tokenizer}-runs.txt"):
                continue

            start = time.time()
            db_name = load_into_db(language=language, tokenizer=tokenizer)
            run_file, average_latency = search_fts5(language=language, tokenizer=tokenizer, db_name=db_name)
            qrels_file = f"/home/hamad/benchmark/pyserini/tools/topics-and-qrels/qrels.miracl-v1.0-{language}-dev.tsv"

            results_file = evaluate(run_file=run_file, qrels_file=qrels_file, average_latency=average_latency)
            end = time.time()
            print(f'Done with {language}-{tokenizer} in {(end-start) // 60} mintues.\n Results saved at {results_file}')
            clean()


if __name__ == '__main__':
    main()