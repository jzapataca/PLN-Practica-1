import re

def process_hashtag(hashtag):

    # eliminar el símbolo de almohadilla
    hashtag = hashtag[1:]

    #verificar si la palabra está en mayúsculas por completo
    if hashtag.isupper():
        pass
    else:
        # separar las palabras con camel case si tienen mas de un digito
        hashtag = re.sub(r"([a-z])([A-Z])", r"\1 \2", hashtag)

    return hashtag


def process_hashtags(text_with_hashtags):

    # encontrar todos los hashtags
    hashtags = re.findall(r"#\w+", text_with_hashtags)

    # procesar cada hashtag
    for hashtag in hashtags:
        processed_hashtag = process_hashtag(hashtag)
        text_with_hashtags = text_with_hashtags.replace(hashtag, processed_hashtag)

    return text_with_hashtags


def preprocess_strings(text):

    # procesar hashtags
    text = process_hashtags(text)

    # convertir a minúsculas
    line = text.lower()

    # eliminar tildes con string translate
    line = line.translate(str.maketrans("áéíóú", "aeiou"))

    # eliminar numeros
    line = ''.join([i for i in line if not i.isdigit()])

    # eliminar urls
    line = re.sub(r"http\S+", "", line)

    # eliminar retornos de carro
    line = line.replace("\r", "")

    # eliminar elementos html con regex
    line = re.sub(r"<.*?>", "", line)

    # eliminar signos de puntuación y emoticones
    line = line.translate(str.maketrans("", "", ".,;:!?¡¿"))

    # cambiar las q con "que"
    line = re.sub(r"\bq\b", "que", line)

    # cambiar las @persona por "usuario"
    line = re.sub(r"@\w+", "usuario:", line)

    # eliminar espacios en blanco
    line = line.strip()

    return line


def preprocess_tweets(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        lines = text.split("\n")[:-1]

    preprocessed_lines = [preprocess_strings(line) for line in lines]

    save_path = file_path.replace("interim", "processed")

    with open(save_path, "w", encoding="utf-8") as file:
        for line in preprocessed_lines:
            file.write(line + "\n")

    print(f"Preprocessed file saved at {save_path}")

    return 1

def preprocess_reviews(file_path):
    "cada review está separada por un Publicada el dd de mes (ejemplo cotubre, o enero)"

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        
        # separar las reviews
        reviews = re.split(r"Publicada el \d{1,2} de \w+", text)
        reviews = reviews[1:]

    preprocessed_reviews = [preprocess_strings(review) for review in reviews]

    print(preprocessed_reviews[0])

# tweet_files = [
#     "./data/interim/pos/tweets.txt",
#     "./data/interim/neg/tweets.txt",
# ]

# for path in tweet_files:
#     preprocess_tweets(path)

preprocess_reviews("./data/interim/neg/neg.txt")