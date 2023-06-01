# cover_lang
An app using langchain and openai to generate a tailored cover letter in seconds

Online Demo: [CoverLang](https://samuelnichols-cover-lang-app-prod-j22e6d.streamlit.app/)

## Running the app locally

install python 3.10+ if you do not have it already
[pyenv](https://github.com/pyenv/pyenv) is a great tool for managing python versions


```bash
pyenv install 3.11.3
```

install all of the required packages

```bash
pip install -r requirements.txt
```

you will need to setup an [openai](https://platform.openai.com/) account and create an [openai api key](https://platform.openai.com/account/api-keys)

use your preferred method to set the environment variable OPENAI_API_KEY to your api key. For example, if you are using bash you can add the following to your .bashrc file

```bash
export OPENAI_API_KEY=your_api_key
```

then just reload your bashrc file

```bash
source ~/.bashrc
```

run the app with streamlit

```bash
streamlit run app.py
```
