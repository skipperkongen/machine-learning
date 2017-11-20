# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
# https://rare-technologies.com/word2vec-tutorial/

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
#import keras

model = gensim.models.KeyedVectors.load_word2vec_format('../models/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
tokenizer = RegexpTokenizer(r"[\w']+")
stopWords = set(stopwords.words('english'))

text = """Loved the original story, had very high expectations for the film (especially since Barker was raving about it in interviews), finally saw it and what can I say? It was a total MESS! The directing is all over the place, the acting was atrocious, the flashy visuals and choreography were just flat, empty and completely unnecessary (whats up with the generic music video techniques like the fast-forward-slow mo nonsense? It was stylish yes but not needed in this film and cheapened the vibe into some dumb MTV Marilyn Manson/Smashing Pumpkins/Placebo music video). Whilst some of the kills are pretty cool and brutal, some are just ridiculously laughable (the first kill on the Japanese girl was hilarious and Ted Raimi's death was just stupidly funny). It just rushes all over the place with zero tension and suspense, totally moving away from the original story and then going back to it in the finale which by that point just feels tacked on to mess it up even more. No explanations were given whatsoever, I mean I knew what was happening only as i'd read the story but for people who hadn't it's even more confusing as at times even i didn't know where it was going and what it was trying to do- it was going on an insane tangent the whole time.<br /><br />God, I really wanted to like this film as i'm a huge fan of Barker's work and loved the story as it has immense potential for a cracking movie, hell I even enjoyed some of Kitamura's movies as fun romps but this film just reeked of amateurism and silliness from start to finish- I didn't care about anyone or anything, the whole thing was rushed and severely cut down from the actual source, turning it into something else entirely. Granted it was gory and Vinnie Jones played a superb badass, but everything else was all over the place, more than disappointing. Gutted"""

for word in tokenizer.tokenize(text):
    if word not in stopWords:
        try:
            vec = model[word]
            print(word, '=>', vec)
        except:
            print(word, 'not in word2vec')
