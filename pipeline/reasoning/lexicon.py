"""Per-language vocabulary and sentence templates for the synthetic
comparative-reasoning dataset (Phase 3, section 3.1).

Every string here is written directly in the target language's script and
grammar -- these are not English templates with names swapped in. Hindi and
Nepali get separate name pools, separate attribute vocabulary, and separate
templates; nothing is shared or transliterated between them.

Comparison words are kept gender/number-invariant on purpose (Hindi
"अधिक"/"कम", Nepali "बढी"/"कम") for the primary templates, so the same
template is grammatical regardless of the grammatical gender of whatever
noun fills the entity slot -- getting Hindi/Nepali adjective-noun gender
agreement right for an arbitrary noun pool programmatically is not
reliable, so the primary templates sidestep it entirely. A handful of
secondary "flavour" templates use attribute-specific adjectives (e.g.
"लंबा" for height) for lexical variety; these are a known simplification
documented in report/phase3_reasoning_report.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attribute:
    key: str
    noun: str          # "height" -> "ऊंचाई"
    unit: str           # "cm" -> "सेंटीमीटर"
    domain: str          # "people" or "objects"
    low: int
    high: int
    poss: str = "की"       # Hindi possessive marker agreeing with `noun`'s grammatical
                            # gender ("X की ऊंचाई" fem. vs "X का वज़न" masc.)
    whose: str = "किसकी"   # same agreement, interrogative form ("किसकी ऊंचाई" vs
                            # "किसका वज़न"); unused by Nepali templates, which use the
                            # gender-invariant "कसको".
    will_be: str = "होगी"   # same agreement, future copula ("अधिक {will_be}" vs "अधिक होगा")


@dataclass
class Lexicon:
    lang: str
    people_names: list[str]
    object_names: list[str]
    attributes: dict[str, Attribute]
    delimiter: str
    yes: str
    no: str
    # template dicts: family -> list of format strings (last one per family
    # is reserved test-only, see generate_dataset.py)
    templates: dict[str, list[str]] = field(default_factory=dict)


HINDI = Lexicon(
    lang="hindi",
    people_names=[
        "राम", "श्याम", "गीता", "सीता", "अर्जुन", "प्रिया", "विकास", "सुनीता",
        "राहुल", "अंजलि", "मोहन", "कविता", "दीपक", "नेहा", "अजय", "पूजा",
        "संजय", "स्वाति", "विनोद", "रेखा", "अमित", "शिवानी", "राजेश", "मीना",
        "सुरेश", "कामिनी", "हरीश", "ललिता", "गोपाल", "रीना", "यश", "भावना",
        "करण", "इशा", "मनोज", "सपना", "विजय", "उर्मिला", "आकाश", "निधि",
    ],
    object_names=[
        "किताब", "बैग", "साइकिल", "मोबाइल", "घड़ी", "कुर्सी", "मेज़", "कलम",
        "जूता", "छाता", "गेंद", "बोतल", "टोपी", "पंखा", "अलमारी", "थैला",
        "चश्मा", "स्कूटर", "टीवी", "जैकेट",
    ],
    attributes={
        "height": Attribute("height", "ऊंचाई", "सेंटीमीटर", "people", 140, 195, poss="की", whose="किसकी", will_be="होगी"),
        "age": Attribute("age", "उम्र", "वर्ष", "people", 5, 80, poss="की", whose="किसकी", will_be="होगी"),
        "price": Attribute("price", "कीमत", "रुपये", "objects", 20, 2000, poss="की", whose="किसकी", will_be="होगी"),
        "weight": Attribute("weight", "वज़न", "किलोग्राम", "objects", 1, 50, poss="का", whose="किसका", will_be="होगा"),
    },
    delimiter="उत्तर:",
    yes="हाँ",
    no="नहीं",
    templates={
        # two-entity numeric comparison, "who has more/less"
        "A_more": [
            "{A} {poss} {attr} {va} {unit} है और {B} {poss} {attr} {vb} {unit} है। {whose} {attr} अधिक है?",
            "{A} {poss} {attr} {va} {unit} है, जबकि {B} {poss} {attr} {vb} {unit} है। {A} और {B} में से {whose} {attr} ज़्यादा है?",
            "यदि {A} {poss} {attr} {va} {unit} और {B} {poss} {attr} {vb} {unit} हो, तो {whose} {attr} अधिक {will_be}?",
        ],
        "A_less": [
            "{A} {poss} {attr} {va} {unit} है और {B} {poss} {attr} {vb} {unit} है। {whose} {attr} कम है?",
            "{A} {poss} {attr} {va} {unit} है, जबकि {B} {poss} {attr} {vb} {unit} है। {A} और {B} में से {whose} {attr} कम है?",
            "यदि {A} {poss} {attr} {va} {unit} और {B} {poss} {attr} {vb} {unit} हो, तो {whose} {attr} कम {will_be}?",
        ],
        # three-entity numeric comparison, superlative
        "B_most": [
            "{A} {poss} {attr} {va} {unit}, {B} {poss} {attr} {vb} {unit}, और {C} {poss} {attr} {vc} {unit} है। तीनों में से {whose} {attr} सबसे अधिक है?",
            "{A}, {B} और {C} {poss} {attr} क्रमशः {va}, {vb} और {vc} {unit} है। सबसे अधिक {attr} {whose} है?",
        ],
        "B_least": [
            "{A} {poss} {attr} {va} {unit}, {B} {poss} {attr} {vb} {unit}, और {C} {poss} {attr} {vc} {unit} है। तीनों में से {whose} {attr} सबसे कम है?",
            "{A}, {B} और {C} {poss} {attr} क्रमशः {va}, {vb} और {vc} {unit} है। सबसे कम {attr} {whose} है?",
        ],
        # pure relational transitive chain (no numbers): A>B, B>C
        "C_endpoints_more": [
            "{A} {poss} {attr}, {B} से अधिक है। {B} {poss} {attr}, {C} से अधिक है। {A} और {C} में से {whose} {attr} अधिक है?",
            "{A}, {attr} में {B} से आगे है। {B}, {attr} में {C} से आगे है। {A} और {C} में से {whose} {attr} अधिक है?",
        ],
        "C_endpoints_less": [
            "{A} {poss} {attr}, {B} से अधिक है। {B} {poss} {attr}, {C} से अधिक है। {A} और {C} में से {whose} {attr} कम है?",
        ],
        "C_most": [
            "{A} {poss} {attr}, {B} से अधिक है। {B} {poss} {attr}, {C} से अधिक है। तीनों में सबसे अधिक {attr} {whose} है?",
        ],
        "C_least": [
            "{A} {poss} {attr}, {B} से अधिक है। {B} {poss} {attr}, {C} से अधिक है। तीनों में सबसे कम {attr} {whose} है?",
        ],
        # equality check, yes/no
        "D_equal": [
            "{A} {poss} {attr} {va} {unit} है और {B} {poss} {attr} {vb} {unit} है। क्या दोनों {poss} {attr} बराबर है?",
        ],
    },
)


NEPALI = Lexicon(
    lang="nepali",
    people_names=[
        "हरि", "गीता", "विष्णु", "सीता", "गोपाल", "सुनिता", "रमेश", "कमला",
        "विकास", "अनिता", "प्रकाश", "सरिता", "दिपक", "माया", "विजय", "सुशीला",
        "नारायण", "इन्दिरा", "कृष्ण", "राधा", "वीरेन्द्र", "सावित्री", "मोहन", "पार्वती",
        "विनोद", "कान्छी", "सन्तोष", "गंगा", "बिनोद", "लक्ष्मी", "राजु", "सुमन",
        "निर्मल", "कल्पना", "शिव", "बिमला", "अर्जुन", "शारदा", "देवी", "टंक",
    ],
    object_names=[
        "किताब", "झोला", "साइकल", "मोबाइल", "घडी", "कुर्सी", "टेबुल", "कलम",
        "जुत्ता", "छाता", "बल", "बोतल", "टोपी", "पंखा", "दराज", "थैलो",
        "चश्मा", "स्कुटर", "टिभी", "ज्याकेट",
    ],
    attributes={
        "height": Attribute("height", "उचाइ", "सेन्टिमिटर", "people", 140, 195),
        "age": Attribute("age", "उमेर", "वर्ष", "people", 5, 80),
        "price": Attribute("price", "मूल्य", "रुपैयाँ", "objects", 20, 2000),
        "weight": Attribute("weight", "तौल", "किलोग्राम", "objects", 1, 50),
    },
    delimiter="जवाफ:",
    yes="हो",
    no="होइन",
    templates={
        "A_more": [
            "{A} को {attr} {va} {unit} छ र {B} को {attr} {vb} {unit} छ। कसको {attr} बढी छ?",
            "{A} को {attr} {va} {unit} छ, जबकि {B} को {attr} {vb} {unit} छ। {A} र {B} मध्ये कसको {attr} बढी छ?",
            "यदि {A} को {attr} {va} {unit} र {B} को {attr} {vb} {unit} भए, कसको {attr} बढी हुन्छ?",
        ],
        "A_less": [
            "{A} को {attr} {va} {unit} छ र {B} को {attr} {vb} {unit} छ। कसको {attr} कम छ?",
            "{A} को {attr} {va} {unit} छ, जबकि {B} को {attr} {vb} {unit} छ। {A} र {B} मध्ये कसको {attr} कम छ?",
            "यदि {A} को {attr} {va} {unit} र {B} को {attr} {vb} {unit} भए, कसको {attr} कम हुन्छ?",
        ],
        "B_most": [
            "{A} को {attr} {va} {unit}, {B} को {attr} {vb} {unit}, र {C} को {attr} {vc} {unit} छ। तीनैमध्ये कसको {attr} सबैभन्दा बढी छ?",
            "{A}, {B} र {C} को {attr} क्रमशः {va}, {vb} र {vc} {unit} छ। सबैभन्दा बढी {attr} कसको छ?",
        ],
        "B_least": [
            "{A} को {attr} {va} {unit}, {B} को {attr} {vb} {unit}, र {C} को {attr} {vc} {unit} छ। तीनैमध्ये कसको {attr} सबैभन्दा कम छ?",
            "{A}, {B} र {C} को {attr} क्रमशः {va}, {vb} र {vc} {unit} छ। सबैभन्दा कम {attr} कसको छ?",
        ],
        "C_endpoints_more": [
            "{A} को {attr}, {B} भन्दा बढी छ। {B} को {attr}, {C} भन्दा बढी छ। {A} र {C} मध्ये कसको {attr} बढी छ?",
            "{A}, {attr} मा {B} भन्दा अगाडि छ। {B}, {attr} मा {C} भन्दा अगाडि छ। {A} र {C} मध्ये कसको {attr} बढी छ?",
        ],
        "C_endpoints_less": [
            "{A} को {attr}, {B} भन्दा बढी छ। {B} को {attr}, {C} भन्दा बढी छ। {A} र {C} मध्ये कसको {attr} कम छ?",
        ],
        "C_most": [
            "{A} को {attr}, {B} भन्दा बढी छ। {B} को {attr}, {C} भन्दा बढी छ। तीनैमध्ये सबैभन्दा बढी {attr} कसको छ?",
        ],
        "C_least": [
            "{A} को {attr}, {B} भन्दा बढी छ। {B} को {attr}, {C} भन्दा बढी छ। तीनैमध्ये सबैभन्दा कम {attr} कसको छ?",
        ],
        "D_equal": [
            "{A} को {attr} {va} {unit} छ र {B} को {attr} {vb} {unit} छ। के दुवैको {attr} बराबर छ?",
        ],
    },
)

LEXICONS = {"hindi": HINDI, "nepali": NEPALI}
