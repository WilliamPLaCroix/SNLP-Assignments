import tokenize
from importlib import reload

from nltk.corpus import webtext


def main():
    import exercise_2  # resolved reload error

    exercise_2 = reload(exercise_2)

    # run on English text by character
    with open(
        "data/full_hsk_sentences.txt", "r", encoding="GBK"
    ) as f:  # resolved UnicodeDecodeError
        exercise_2.analysis("Mandarin by character", list(f.read().lower()))

    # run on German text by character
    with open(
        "data/macbeth_de.txt", "r", encoding="utf-8"
    ) as f:  # resolved UnicodeDecodeError
        exercise_2.analysis("German by character", list(f.read().lower()))

    # run on PIRATES OF THE CARRIBEAN: DEAD MAN'S CHEST by character
    text = str(webtext.raw("pirates.txt"))
    exercise_2.analysis("Pirates by character", list(text.lower()))
    # and call the function as done above

    # Run on Transformer's trainer module's source code by character
    with open("data/trainer.py", "r", encoding="utf-8") as f:  # resolved linter warning
        exercise_2.analysis("Python by character", list(f.read().lower()))

    # run on English text no lower()
    with open(
        "data/full_hsk_sentences.txt", "r", encoding="GBK"
    ) as f:  # resolved UnicodeDecodeError
        exercise_2.analysis("English no lower()", f.read().split())

    # run on German text no lower()
    with open(
        "data/macbeth_de.txt", "r", encoding="utf-8"
    ) as f:  # resolved UnicodeDecodeError
        exercise_2.analysis("German no lower()", f.read().split())

    # run on PIRATES OF THE CARRIBEAN: DEAD MAN'S CHEST no lower()
    text = str(webtext.raw("pirates.txt"))
    exercise_2.analysis("Pirates no lower()", list(text.split()))
    # and call the function as done above

    # Run on Transformer's trainer module's source code no lower()
    with open("data/trainer.py", "r", encoding="utf-8") as f:
        tokens = [
            x.string
            for x in tokenize.generate_tokens(f.readline)
            if x.type
            not in {
                tokenize.COMMENT,
                tokenize.STRING,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
            }
        ]
    exercise_2.analysis("Python no lower()", tokens)


if __name__ == "__main__":
    main()
