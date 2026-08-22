"""
make_fixture.py — synthetic raw documents for offline pipeline testing.

Exercises build_corpus -> tokenizer -> count -> report without touching the
network. Deliberately includes near-duplicates, wrong-language documents,
boilerplate and junk so each filter has something to catch.
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.manifest import Document, ShardWriter

random.seed(17)

HI_ROOT = ("सरकार योजना किसान आर्थिक सहायता कृषि मंत्री ग्रामीण अर्थव्यवस्था व्यापार शिक्षा "
           "स्वास्थ्य न्यायालय संसद विधानसभा पुलिस अस्पताल विद्यालय पुस्तकालय कार्यालय "
           "अधिकारी कर्मचारी नागरिक समाज संस्कृति साहित्य संगीत विज्ञान उद्योग निर्माण "
           "परिवहन ऊर्जा पर्यावरण नदी पर्वत बाजार मुद्रा बैंक निवेश").split()
HI_SUF = ["", "ों", "ी", "ें", "का", "के", "की", "में", "से", "पर", "को", "ने", "ता"]
HI_V = "है हैं था थे किया गया रहा बताया कहा दिया नहीं लेकिन".split()

NE_ROOT = ("सरकार कार्यक्रम किसान आर्थिक सहयोग कृषि मन्त्री ग्रामीण अर्थतन्त्र व्यापार शिक्षा "
           "स्वास्थ्य अदालत संसद प्रदेश प्रहरी अस्पताल विद्यालय पुस्तकालय कार्यालय "
           "कर्मचारी नागरिक समाज संस्कृति साहित्य संगीत विज्ञान उद्योग निर्माण "
           "यातायात ऊर्जा वातावरण नदी हिमाल बजार मुद्रा बैंक लगानी").split()
NE_SUF = ["", "हरू", "ले", "लाई", "मा", "बाट", "सँग", "को", "का", "की", "सम्म"]
NE_V = "छ छन् थियो थिए गरेको भएको हुनेछ बताउनुभयो भन्ने गर्ने".split()


def vocab(roots, sufs, n):
    combos = [r + s for r, s in itertools.product(roots, sufs)]
    random.shuffle(combos)
    return combos[:n]


def prose(words, verbs, n_sent=None):
    n_sent = n_sent or random.randint(10, 28)
    return "। ".join(
        " ".join(random.choice(words) for _ in range(random.randint(8, 24)))
        + " " + random.choice(verbs) for _ in range(n_sent)) + "।"


def build(lang, roots, sufs, verbs, other_words, other_verbs, root: Path):
    words = vocab(roots, sufs, 2500)
    plan = [
        ("downloaded", "sangraha", "hf_download", 900),
        ("downloaded", "indiccorp", "hf_download", 500),
        ("downloaded", "wikipedia", "hf_download", 200),
        ("manual", "example-news.com", "scrape", 380),
        ("manual", "ocr:book_1", "ocr", 170),
    ]
    for cls, src, method, n in plan:
        w = ShardWriter(root / lang / "data" / "raw" /
                        f"{'downloaded_' if cls == 'downloaded' else 'manual_'}{src.replace(':', '_').replace('.', '_')}.jsonl",
                        resume=False)
        made = 0
        while made < n:
            r = random.random()
            if r < 0.06:                      # near-duplicate of the previous doc
                text = prose(words, verbs, 14)
                w.write(Document(text=text, language=lang, provenance_class=cls,
                                 source=src, collection_method=method))
                text2 = text + " यह अतिरिक्त वाक्य है।"
                w.write(Document(text=text2, language=lang, provenance_class=cls,
                                 source=src, collection_method=method))
                made += 2
                continue
            if r < 0.10:                      # wrong language
                text = prose(other_words, other_verbs, 12)
            elif r < 0.14:                    # junk / boilerplate
                text = "यह भी पढ़ें\n" * 40
            elif r < 0.17:                    # too short
                text = prose(words, verbs, 1)
            else:
                text = prose(words, verbs)
                if random.random() < 0.3:
                    text = "यह भी पढ़ें\n" + text + "\nसर्वाधिकार सुरक्षित\n42"
            w.write(Document(text=text, language=lang, provenance_class=cls,
                             source=src, collection_method=method))
            made += 1
        w.close()
        print(f"  {lang}/{src}: {w.written} written")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    hi_w = vocab(HI_ROOT, HI_SUF, 2500)
    ne_w = vocab(NE_ROOT, NE_SUF, 2500)
    build("hindi", HI_ROOT, HI_SUF, HI_V, ne_w, NE_V, root)
    build("nepali", NE_ROOT, NE_SUF, NE_V, hi_w, HI_V, root)
    print("fixture ready")
