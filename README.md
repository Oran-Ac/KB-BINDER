# KB-BINDER
The implementation for paper [Few-shot In-context Learning for Knowledge Base Question Answering](http://arxiv.org/abs/2305.01750)
<img width="1237" alt="KBQA-BINDER" src="https://github.com/ltl3A87/KB-BINDER/assets/55973524/d9ceefbe-392e-4749-bf1f-58a93e97b254">

## Set up

1. Set up the knowledge base server: Follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. After starting your virtuoso service, replace the url in `sparql_executer.py` with your own.
```shell
$ bash setup_freebase.sh # it took about 50min to download the data(about 53GB)
$ bash setup_virtuoso.sh 
# Running the server
$ python3 virtuoso/virtuoso.py start 3001 -d ./virtuoso
# Stopping the server
$ python3 virtuoso/virtuoso.py stop 3001
```
2. Download GrailQA/WebQSP/GraphQA/MetaQA and other required files from the [link](https://drive.google.com/drive/folders/1g8ZpMLSw95KwjisXEw07rVVC3TJ1LZdn?usp=sharing) and put them under `data/`.
3. Install all required libraries:
```
# if you are using HPC
$ module load gcc/13.2.0  
# if you are using local machine
$ sudo apt-get update
$ sudo apt-get install build-essential g++
# then
$ pip install -r requirements.txt
```
You can download the index file and put it under `contriever_fb_relation
/freebase_contriever_index/` with this [link](https://drive.google.com/file/d/1hnyW-_k0YaAUZDTdYzhbKDTnFuLEW-W2/view?usp=sharing).

4. Use our LLM with the openai API coding style.
```shell
# install the package that we need
$ pip install "fschat[model_worker,webui]" # refer to: https://github.com/lm-sys/FastChat/tree/main
```
If you are with the local machine, you can run the following command to start the controller, model worker, and RESTful API server.
```shell
# launch the controller
$ python -m fastchat.serve.controller
# launch the worker
$ python -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
# launch the RESTful API server
$ python -m fastchat.serve.openai_api_server --host localhost --port 8000
```
If you are with the HPC, you can run the following command to start the controller, model worker, and RESTful API server.
```shell
# Undo
```
## Run Experiments

### KB-BINDER:
```
$ python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key [your api key] --engine [engine model name] \
 --train_data_path [your train data path] --eva_data_path [your eva data path] \
 --fb_roles_path [your freebase roles file path] --surface_map_path [your surface map file path]
```

### KB-BINDER-R:
```
$ python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key [your api key] --engine [engine model name] --retrieval \
 --train_data_path [your train data path] --eva_data_path [your eva data path] \
 --fb_roles_path [your freebase roles file path] --surface_map_path [your surface map file path]
```


As the codex API has been closed, you may use other engine.
