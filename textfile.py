import re
class Textfile:

    def __init__(self, text):
        self.text = str(text)

    def __str__(self):
        return self.text

    def remove_special_characters(self, chars_to_remove=[], lowercase=True):
        chars_to_remove = '[' + re.escape(''.join(chars_to_remove)) + ']'
        text = re.sub(chars_to_remove, '', self.text) + " "
        if (lowercase):
            text = text.lower() + " "
        return Textfile(text)

    # mit mario reden.... ä ü ö tragen reichhaltige phnoetische information also drin lassen.... wie ist das mit ß/ss, groß/klein (zumindest phonetisch irrelevant)
    def replace_special_characters(self, chars_to_replace=['ß'], replacement=["ss"]):
        text = self.text
        for i, char in enumerate(chars_to_replace):
            text = re.sub(char, replacement[i], text)
        return Textfile(text)

    def preprocess(self, chars_to_remove, chars_to_replace, replacement, lowercase=True):
        text_removed = self.remove_special_characters(chars_to_remove)
        text_clean = text_removed.replace_special_characters(chars_to_replace, replacement)
        return Textfile(text_clean)




