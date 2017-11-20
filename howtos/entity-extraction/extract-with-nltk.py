import nltk
from nltk import ne_chunk, pos_tag, word_tokenize

text = """
The European Convention on Human Rights has played an important role in the development and awareness of Human Rights in Europe. The development of a regional system of human rights protection operating across Europe can be seen as a direct response to twin concerns. First, in the aftermath of the Second World War, the convention, drawing on the inspiration of the Universal Declaration of Human Rights can be seen as part of a wider response of the Allied Powers in delivering a human rights agenda through which it was believed that the most serious human rights violations which had occurred during the Second World War could be avoided in the future. Second, the Convention was a response to the growth of Communism in Central and Eastern Europe and designed to protect the member states of the Council of Europe from communist subversion. This, in part, explains the constant references to values and principles that are "necessary in a democratic society" throughout the Convention, despite the fact that such principles are not in any way defined within the convention itself.[6]

From 7 to 10 May 1948 with the attendance of politicians (such as Winston Churchill, François Mitterrand and Konrad Adenauer), civil society representatives, academics, business leaders, trade unionist and religious leader was organised gathering-The "Congress of Europe" in Hague. At the end of Congress the declaration and following pledge was issued which demonstrated the initial seeds of modern European institutes, including ECHR. The second and third Articles of Pledge stated: We desire a Charter of Human Rights guaranteeing liberty of thought, assembly and expression as well as right to form a political opposition. We desire a Court of Justice with adequate sanctions for the implementation of this Charter.[7]
"""


for sent in nltk.sent_tokenize(text):
    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            print ('Label={}  Leaves={}'.format(chunk.label(), chunk.leaves()))
