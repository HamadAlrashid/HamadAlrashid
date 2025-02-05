# Hi there, I'm Hamad Alrashid👋

Passionate, self-motivated, and detail-oriented undergraduate student who loves to explore new technologies and build innovative projects. I study Computer Science at the University of Maryland ([Ranks top 10 in CS](https://cmns.umd.edu/news-events/news/computer-science-undergraduate-program-ranks-top-10)) with an academic emphasis in ML/AI, Data Science, and Cybersecurity. 


### Table of Contents
1. [Work Experience](#0)
2. [Project 1: Robust Data & AI Pipeline (Instagram API) at Ario](#1)
3. [Project 2: ML Tokenization in lexical-matching Retrieval Systems at Vectara.com](#2)
4. [Project 3: Computer Olympiad (Cryptography Web Application) ](#3)
5. [Contact Information](#4)

## Work Experience <a name="0"></a>

- **AI & Machine Learning Intern** (2024 - Palo Alto) 
  - During my 2024 Summer internship at [Ario](https://www.heyario.com/), a Silicon Valley-based AI startup with $16 million in seed funding, I
    contributed to a dynamic high-paced, competitive environment by working on various SWE & AI projects, mainly vertically developing a robust data pipeline for integrating a new data source, and thus, enhancing the application's data richness and functionality. [Blog Post](https://www.linkedin.com/posts/heyario_dataengineering-datapipeline-startups-ugcPost-7259961925860532224-0_mq)


- **Software Engineer Intern** (2023 - Palo Alto)
  - At Vectara, a top-50 GenAI startup, in Silicon Valley with a $55 million series-A funding, I worked on a 3-month cutting-edge project that improves the Information Retrieval System using Machine
Learning. Performed extensive ML experiments & IR QA testing (Pyserini/Anserini, FTS5) & hyper-parameter tuning (Google Vizier) for BM25, outperforming strong baselines.


- **Research Intern** at University of California, Irvine (UCI) (2022 - Irvine)
  - Under Dr.William Rangwei Mao's supervision, I researched developing a wireless endoscope for medical purposes. Wrote C# and Python scripts for transmitting and encoding/decoding video data using JPEG compression algorithm.

## 🚀 Project 1: Robust Data & AI Pipeline (Instagram API) at Ario <a name="1"></a>

### Description
During my 2024 AI & ML internship at Ario, I fully implemented a robust data pipeline for integrating Instagram as a new data source in the company's main application. The pipeline handles the complete flow from user authentication with Instagram to data processing and storage, utilizing the official Instagram Basic Display API to authorize Instagram and fetch user posts.

### Goals of the project
- Goal 1: Implement a complete data pipeline for Instagram integration
- Goal 2: Handle secure user authentication with Instagram
- Goal 3: Develop robust data fetching & normalization processes
- Goal 4: Design efficient data compression and storage solutions
- Goal 5: Create a modular and extensible social media processor framework
- Goal 6: Extensive LLM evaluation and prompt tuning
- Goal 7: Enrich the application's data ecosystem

### Technologies/Resources Used
- Instagram Basic Display API: Official API for Instagram authorization and data fetching
- Flask: Development server for handling authentication with the API
- Pydantic: Data validation and settings management
- Facebook Developer Platform: App creation and API configuration
- Environment Variables: Secure configuration management
- Database Integration: Data models and custom CRUD operations
- LLM Integration: Custom prompts for data processing
- SQLAlchemy, Multithreading...

### Key Components
- Abstract Base Class: Modular framework for implementing social media processors
- Instagram Processor: Complete implementation of Instagram data processing
- API Integration: Direct interaction with Instagram Basic Display API
- Authentication Flow: Auth implementation with Flask
- Data Validation: Robust JSON validation using Pydantic
- Database Operations: Efficient CRUD functionality
- Data Models: Well-structured data objects and types
- LLM Prompts: Well-written & tested prompts for data enrichment

### Links
- 🔗 [Project Repository](https://github.com/HamadAlrashid/instagram_basic_display_api/)


## 🚀 Project 2: Using Machine Learning Tokenization in lexical-matching Retrieval Systems at [Vectara.com](https://vectara.com/), a top-50 GenAI startup, at the Silicon Valley. <a name="2"></a>

### Description
During my 2023 summer internship at Vectara.com, I worked on a 3-month cutting-edge project that improved the Information Retrieval system using Machine Learning. I performed extensive ML experiments & Information Retrieval question-answering benchmarking & Hyper-Parameter tuning, outperforming strong baselines such as the mBERT tokenizer. 
*LinkedIn post: [Post](https://www.linkedin.com/posts/hamad-alrashid-3a94bb142_%D8%A7%D9%84%D8%AD%D9%85%D8%AF%D9%84%D9%84%D9%87-im-excited-to-announce-that-ive-activity-7101650295583121408-g--0)


### Goals of the project
- Goal 1: Understanding the importance of tokenization in lexical-matching retrieval systems  
- Goal 2: Extensive analysis of the [Sentencepiece](https://github.com/google/sentencepiece) ML tokenizer (Google, 2018) in BM25 
- Goal 3: Hyper-parameter tuning experiments
- Goal 4: Train a Multilingual Tokenizer that supports 41 languages
- Goal 5: Expand on [Better Than Whitespace: Information Retrieval for Languages without Custom Tokenizers](https://arxiv.org/abs/2210.05481), a paper by Jimmy Lin's research group at the University of Waterloo
- Goal 6: Improve the effectiveness of the keyword retrieval system
- And more...

### Technologies/Resources Used
- [Sentencepiece](https://github.com/google/sentencepiece): An open-source Machine Learning Tokenizer, developed and maintained by Google. 
- BM25 / TF-IDF: keyword-matching algorithm
- [Pyserini](https://github.com/castorini/pyserini): "toolkit for reproducible information retrieval research with sparse and dense representations"
- Python: Used for writing the benchmarking & training scripts
- [Mr. TiDy](https://aclanthology.org/2021.mrl-1.12.pdf) & [MIRACL](https://github.com/project-miracl/miracl): Multilingual QA datasets for benchmarking retrieval systems
- [Wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b): Wikipedia text of over 84 million sentences of 41 languages
- [Google Vizier](https://cloud.google.com/vertex-ai/docs/vizier/overview): A black-box optimizer that is used for hyper-parameter tuning
- Amazon EC2 Instance: Used a machine with over 700GB of RAM to train the tokenizer
- [SQLITE Full-Text search](https://www.sqlite.org/fts5.html): A powerful extension that provides a full-text search engines to efficiently search a corpus

### Some of the results:
- Outperforming the mBERT tokenizer and closing the gap with language-specific tokenizers (Language analyzers)
![One of the benchmarks against strong baselines](/Sentencepiece/Recall.PNG?raw=true "Recall @ 100")
![One of the benchmarks against strong baselines2](/Sentencepiece/All.PNG?raw=true "ndcg @ 10")
- Hyperparameter tuning (500+ tokenizer experiments for 11+ Langauges). English:
![Hyperparameter tuning](/Sentencepiece/English.PNG?raw=true "Hyperparameter tuning")


### Links for some of the scripts I wrote
- 🔗 [Benchmark BM25 using Pyserini (Lucene)](Sentencepiece/benchmark_pyserini.py)
- 🔗 [Benchmark BM25 on SQLITE (fts5)](Sentencepiece/benchmark_fts5.py)


## 🌟 Project 3: Cryptography Web Application <a name="3"></a>
* Won the second-best project in the National Computer Olympiad in 2018 in Saudi Arabia in Riyadh


### Description
A web application for educating users about encryption and hash algorithms in the Arabic language. I noticed a huge gap in the Arabic resources in the field of cryptography. Therefore, I developed this project to enrich the Arabic resources and showcase how encryption & hashing algorithms work. I submitted my project to the 2018 National Computer Olympiad In Riyadh, where I won the second-best project.

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
- OpenSSL library
- Bootstrap for UI

### Some pages of the project
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/1.PNG)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/2.png)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/3.PNG)
![Project 2 Screenshot](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-/blob/main/4.PNG)


### Links
- 🔗 [Full project](https://github.com/HamadAlrashid/Encryption-web-application-Arabic-)


## Let's Connect! <a name="4"></a>

Feel free to reach out if you have any questions

- 📫 Email: hamad1 AT umd.edu
- :briefcase: [LinkedIn](https://www.linkedin.com/in/hamad-alrashid-3a94bb142/) 
- :page_with_curl: [My Resume](https://drive.google.com/file/d/14eG4jlyFKsUgzrR6ztd1VnNnw8ZjQQL8/view?usp=sharing)



