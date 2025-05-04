from scripts.text_preprocessing import EnglishProcessor


class EnglishArgs:
    remove_urls: bool = True
    remove_digits: bool = True
    remove_emails: bool = True
    remove_stopwords: bool = False
    remove_punctuations: bool = True
    remove_html: bool = True
    use_lemmatizer: bool = False
    use_stemmer: bool = False


eng_processor = EnglishProcessor(arguments = EnglishArgs)