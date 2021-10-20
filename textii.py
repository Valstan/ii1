import pymorphy2
import re
import pandas as pd

morph = pymorphy2.MorphAnalyzer(lang='ru')
stmt = "SELECT * FROM {database}.{table}".format(database=db['database'], table=db['table'])

df = pd.read_sql(stmt, conn)




def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 3)
    text = text.encode("utf-8")

    return text


df['Description'] = df.apply(lambda x: clean_text(x[u'Описание заявки']), axis=1)
