## Real-time Measurement of Reasoning in Meetings

This repository is set up to be deployed to Azure as an **Online Managed Endpoint for real-time inferencing** (see https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online).
The minimum recommended compute SKU is the one listed in the deployment files already. 
Theoretically you can run the code locally as well, depending on compute power - memory in particular, as all models and transcripts are stored in memory.

Before running or deploying the system, you need the following HuggingFace models cloned. Run the following in a terminal from within the 'models' subfolder:
- `git clone https://huggingface.co/HCKLab/BiBert-Subjectivity`
- `git clone https://huggingface.co/GroNLP/mdebertav3-subjectivity-multilingual`
- `git clone https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768`
- `git clone https://huggingface.co/pedropei/sentence-level-certainty`
- `git clone https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment`

Once you are ready to deploy, you can do so with the following commands, provided you have set up all necessary Azure resources:
- az ml online-endpoint create -n mc-reasoning -f endpoint.yml
- az ml online-deployment create -n reasoning-paper -f deployment.yml --all-traffic

Inference calls to the endpoint should be in the following format, with mcid being a identifier unique to the overall text you are sending new parts off for live inferencing. 

`{"text": "", "mcid": ""}`
