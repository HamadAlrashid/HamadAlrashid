# Hi there, I'm Hamad Alrashid👋

Passionate, self-motivated, and detail-oriented undergraduate student who loves to explore new technologies and build innovative projects. I study Computer Science at the University of Maryland ([Ranks top 10 in CS](https://cmns.umd.edu/news-events/news/computer-science-undergraduate-program-ranks-top-10)) with academic emphasis in ML/AI, Data Science, and Cybersecurity. Below are two of my favorite projects that I've worked on. 


### Table of Contents

1. [Project 1: ML Tokenization in lexical-matching Retrieval Systems at Vectara.com](#1)
2. [Project 2: Cryptography Web Application](#2)
3. [Contact Information](#3)

## 🚀 Project 1: Using Machine Learning Tokenization in lexical-matching Retrieval Systems at [Vectara.com](https://vectara.com/), a top-50 GenAI startup, at the Silicon Valley. <a name="1"></a>

### Description
During my 2023 summer internship at Vectara.com, I worked on a 3-month cutting-edge project that improves the Information Retrieval system using Machine Learning. I performed extensive ML experiments & Information Retrieval Question-Answering benchmarking & Hyper-Parameter tuning, outperforming strong baselines such as the mBERT tokenizer. 
*LinkedIn post: [Post](https://www.linkedin.com/posts/hamad-alrashid-3a94bb142_%D8%A7%D9%84%D8%AD%D9%85%D8%AF%D9%84%D9%84%D9%87-im-excited-to-announce-that-ive-activity-7101650295583121408-g--0)


### Goals of the project
- Goal 1: Understanding the importance of tokenization in lexical-matching retrieval systems  
- Goal 2: Extensive analysis of the [Sentencepiece](https://github.com/google/sentencepiece) ML tokenizer (Google, 2018) in BM25 
- Goal 3: Hyper-parameter tuning experiments
- Goal 4: Train a Multilingual Tokenizer that supports 41 languages
- Goal 5: Expand on [Better Than Whitespace: Information Retrieval for Languages without Custom Tokenizers](https://arxiv.org/abs/2210.05481) , a paper by Jimmy Lin's research group at the University of Waterloo
- Goal 6: Improve the effectiveness of the keyword retrieval system
- And more...

### Technologies/Resources Used
- [Sentencepiece](https://github.com/google/sentencepiece) : An open-source Machine Learning Tokenizer, developed and maintained by Google. 
- BM25 / TF-IDF: keyword-matching algorithm
- [Pyserini](https://github.com/castorini/pyserini): "toolkit for reproducible information retrieval research with sparse and dense representations"
- Python: Used for writing the benchmarking & training scripts
- [Mr. TiDy](https://aclanthology.org/2021.mrl-1.12.pdf) & [MIRACL](https://github.com/project-miracl/miracl): Mutlilingual QA datasets for benchmarking retrieval systems
- [Wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b): Wikipedia text of over 84 million sentences of 41 langauges
- [Google Vizier](https://cloud.google.com/vertex-ai/docs/vizier/overview): A black-box optimizer that is used for hyper-parameter tuning
- Amazon EC2 Instance: Used a machine with over 700GB of RAM to train the tokenizer
- [SQLITE Full-Text search](https://www.sqlite.org/fts5.html): A powerful extension that provides a full-text search engines to efficiently search a corpus

### Some of the results:
- Outperforming the mBERT tokenizer and closing the gap with langauge-specific tokenizers (Language analyzers)
![One of the benchmarks against strong baselines](/Sentencepiece/Recall.PNG?raw=true "Recall @ 100")
![One of the benchmarks against strong baselines2](/Sentencepiece/All.PNG?raw=true "ndcg @ 10")
- Hyperparameter tuning (500+ tokenizer experiments for 11+ Langauges). English:
![Hyperparameter tuning](/Sentencepiece/English.PNG?raw=true "Hyperparameter tuning")


### Links for some of the scripts I wrote
- 🔗 [Benchmark BM25 using Pyserini (Lucene)](Sentencepiece/benchmark_pyserini.py)
- 🔗 [Benchmark BM25 on SQLITE (fts5)](Sentencepiece/benchmark_fts5.py)


## 🌟 Project 2: Cryptography Web Application <a name="2"></a>
*Won the second best project in the National Computer Olympiad in 2018 in Saudi Arabia in Riyadh


### Description
A web-application for educating users about encryption and hash algorithms in the Arabic language. I noticed a huge gap in the Arabic resources in the field of cryptography. Therefore, I developed this project to enrich the Arabic resources and showcase how encryption & hashing algorithms work. I submitted my project to the 2018 National Computer Olympiad In Riyadh, where I won the second best project.

### Features
- Feature 1: Supports DarkMode  
- Feature 2: Ceasar Cipher
- Feature 3: AES
- Feature 4: hex to string & string to hex
- Feature 5: base64 encoder/decoder
- Feature 6: MD5
- Feature 7: SHA1
- Feature 8: SHA256

### Technologies Used
- PHP
- Xampp
- Openssl library
- Bootstrap for UI

### Some pages of the project
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/1.PNG)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/2.png)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/3.PNG)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/4.PNG)


### Links
- 🔗 [Full project](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-)


## Let's Connect! <a name="3"></a>

Feel free to reach out if you have any questions

- 📫 Email: hamad1 AT umd.edu
- :briefcase: [LinkedIn](https://www.linkedin.com/in/hamad-alrashid-3a94bb142/) 
- :page_with_curl: [My Resume](https://drive.google.com/file/d/16el7m3PX742xl6r_G5O0BnBKWGc1zS40/view?usp=sharing)



