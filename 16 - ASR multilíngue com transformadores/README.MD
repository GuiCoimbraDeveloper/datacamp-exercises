No modelo ASR multilíngue, você ajustará o modelo Wave2Vec XLS-R usando áudio e transcrição turcos para criar um sistema de reconhecimento automático de fala. 

Primeiro, você entenderá os arquivos de áudio e o conjunto de dados de texto e, em seguida, usará um tokenizador de texto, extrairá recursos e processará os arquivos de áudio. Depois disso, você criará um treinador, uma função WER, carregará modelos pré-treinados, ajustará hiperparâmetros, treinará e avaliará o modelo. 

Você pode usar a plataforma Hugging Face para armazenar os pesos do modelo e publicar aplicativos da Web para transcrever a fala em tempo real: Transmissão de Urdu Asr.

https://huggingface.co/blog/fine-tune-xlsr-wav2vec2