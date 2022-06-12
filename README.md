# Real Time News Assistant

Real Time News Assistant for greek news.

## How to run the project
First of all it is advised to use a python virtual environment to run this project. The version of python used was 3.8.10

### Create Virtual Environment
```
python3 -m venv /path/to/new/virtual/environment
```
Make sure the virtual environment is enabled and then run:
```
pip3 install -r requirements.txt
```
This will install all the required python modules.

### Set up the database
In order to set up the elasticsearch database you can use haystacks abstraction that creates and runs a docker container with the default configuration. In a python script just run:
```
from haystack.utils import launch_es
launch_es()
```
This will create a docker container with an elasticsearch database running at http://localhost:9200

You can then add documents using the API provided in the QASystem as follows:
```
db = Database()
db.connect()
db.add_documents(docs)
```
Where docs is a list of dictionaries where its dictionary is a document. To connect with custom creadentials or to a different port check the API. After the initialization of the database we can run the action server.

For more customization of the database you will need to set up the container as described [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
### Run the Action server
In order to run the action server open a terminal in the rasa folder with the venv activated and run:
```
rasa run actions
```
Make sure the actions endpoint is uncommented in the endpoint.yml file

### Set up RASA-X
In order to set up RASA-X the RASA Ephemeral Installer was used. More details are provided [here](https://github.com/RasaHQ/REI).
Run:
```
bash rei.sh -y
rasactl start rasa-x --values-file values.yml
```
Also make sure you have set up the existingUrl in the values.yml file to the endpoint of the action server. You will need to port forward the IP of the action server in order for the RASA-X to be able to access the action server we set up earlier.

The assistant can be trained and activated using RASA-X.

## Reference

```
@article{Artetxe:etal:2019,
  author        = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
  title         = {On the cross-lingual transferability of monolingual representations},
  journal       = {CoRR},
  volume        = {abs/1910.11856},
  year          = {2019},
  archivePrefix = {arXiv},
  eprint        = {1910.11856}
}
```
```
@Article{Devlin2019,
  author        = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal       = {arXiv:1810.04805 [cs]},
  title         = {{BERT}: {Pre}-training of {Deep} {Bidirectional} {Transformers} for {Language} {Understanding}},
  year          = {2019},
  month         = may,
  note          = {arXiv: 1810.04805},
  abstract      = {We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5\% (7.7\% point absolute improvement), MultiNLI accuracy to 86.7\% (4.6\% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).},
  file          = {:Devlin2019 - BERT_ Pre Training of Deep Bidirectional Transformers for Language Understanding.pdf:PDF},
  keywords      = {Computer Science - Computation and Language},
  priority      = {prio1},
  readstatus    = {read},
  shorttitle    = {{BERT}},
  url           = {http://arxiv.org/abs/1810.04805},
  urldate       = {2021-12-09},
}
```
```
 @misc{rasa_2021,
  title         = {Open source conversational AI},
  url           = {https://rasa.com/},
  journal       = {Rasa},
  year          = {2021},
  month         = {Nov},
  note          = {Accessed: 2022-04-28}
}
```
```
 @misc{haystack docs,
  title         = {Haystack docs},
  url           = {https://haystack.deepset.ai/},
  journal       = {Haystack Docs},
  year          = {2021},
  month         = {Dec},
  note          = {Accessed: 2022-04-28}
}
```
