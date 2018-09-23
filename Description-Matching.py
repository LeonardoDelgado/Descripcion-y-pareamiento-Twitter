from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re,string
from nltk.corpus import stopwords
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity

def elimstopwords(data):#,hashtag): #preprosesado
    data = unicodedata.normalize("NFKD", data).encode("ascii","ignore").decode("ascii")
    words = re.findall(r"[\w]+|[^\s\w]", data.lower(), re.UNICODE) 
    etiqueta = words[0]
    important_words = []
    for word in words:
        if word not in stopwords.words('spanish') and len(word)>3: #and word not in hashtag:
            important_words.append(word)
    return ''.join(x + ' ' for x in important_words), etiqueta.upper()

def rasgos(data, n_samples = 20, n_features = 20): #Extraccion de ragos
    data = [data]
    tf_vectorizer = CountVectorizer(min_df = 0.01,
                                max_features = n_features,
                                strip_accents = 'ascii',
                                analyzer = 'word'
                                 )
    tf = np.array(tf_vectorizer.fit_transform(data).toarray())
    palabras = tf_vectorizer.vocabulary_
    histograma = tf 
    his = unir(palabras,histograma)
    return his

def unir(palabras,histograma):
    his = {}
    for palabra in palabras:
        his[palabra] = histograma[0,palabras[palabra]]    
    return his

def distancia(elemento1,elemento2):
    dis=0
    for palabra in elemento1:
        if palabra in elemento2:
            dis += elemento1[palabra]/elemento2[palabra]
    return dis

def distanciacos(elemento1,elemento2):
    a = set(elemento1.keys())
    b = set(elemento2.keys())
    claves = list(a.union(b))
    vec1 = []
    vec2 = []
    for clave in claves:
        if clave in elemento1:
            vec1.append(elemento1[clave])
        else:
            vec1.append(0)
        if clave in elemento2:
            vec2.append(elemento2[clave])
        else:
            vec2.append(0)
    return float(cosine_similarity(np.array(vec1).reshape(1, -1),np.array(vec2).reshape(1, -1)))

def match(base_de_conocimiento,prueba,dist = 1):
    dis =[]
    for elemento in base_de_conocimiento:
        if dist ==1:
            dis.append(distancia(prueba,elemento))
        else:
            dis.append(distanciacos(prueba,elemento))
            
    return dis

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
    
def get_conocimiento(data_samples, retiquetas = False):
    datos = []
    etiquetas = []
    caracteristicas = []
    for data in data_samples:
        data = strip_all_entities(data)
        texto,etiqueta = elimstopwords(data)
        datos.append(texto)
        etiquetas.append(etiqueta)
        caracteristicas.append(rasgos(texto))
    if retiquetas:
        return caracteristicas, etiquetas
    else:
        return caracteristicas


if '__main__' == __name__:
    
    print("Loading dataset...")
    file_train = open('corpus twitter.txt','r')
    data_samples=file_train.readlines()
    file_train.close()
    base_de_conocimiento = get_conocimiento(data_samples)
    
    print("Loading dataset...")
    file_train = open('corpus twitter prueba.txt','r')
    data_samples=file_train.readlines()
    file_train.close()
    pruebas = get_conocimiento(data_samples)
    dis = []
    dis1 = []
    for prueba in pruebas:
        dis.append(match(base_de_conocimiento,prueba))
        dis1.append(match(base_de_conocimiento,prueba, dist=2))
    dis = np.array(dis)
    dis1 = np.array(dis1)
        




